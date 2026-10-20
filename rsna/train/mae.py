"""MAE over the SSL plane cache, continued from DINOv2 ViT-S/14.

Encoder facts verified before writing (timm vit_small_patch14_dinov2 @336):
    embed_dim 384 | 12 blocks | num_prefix_tokens 1 (cls, NO registers)
    patch_embed 3->384, 14x14 stride 14 | grid 24x24 = 576 patches | pos_embed (1,577,384)
    norm=LayerNorm, fc_norm=Identity  (the layout load_for_inference expects)
So our 2.5D triplet feeds the existing 3-channel patch embedding unchanged, and masking is
inherently JOINT across the three channels because a patch spans all of them.

Sampling: study -> series -> plane, each uniform. Uniform sampling over planes would weight
long series and multi-series studies more heavily (review, section on corpus definition).
No flips: the cache is already canonicalised (deterministic mirror); random mirroring would
undo that and is banned for the same reason hflip is banned downstream.
"""
from __future__ import annotations
import random
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]
SSL_DIR = ROOT / "ssl_cache_336_130mm/train"
INDEX = ROOT / "artifacts/ssl_index.parquet"
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def build_index(force: bool = False) -> pd.DataFrame:
    """(study, uid, n_planes) for every cached series. mmap so this reads headers, not pixels."""
    n_cached = len(list(SSL_DIR.glob("*.pt")))
    if INDEX.exists() and not force:
        df = pd.read_parquet(INDEX)
        # A STALE index is a silent catastrophe: it would pretrain on the QA subset while
        # reporting success. Invalidate whenever the cache no longer matches the index.
        if df.study.nunique() == n_cached:
            return df
        print(f"index stale ({df.study.nunique()} studies indexed vs {n_cached} cached) -- rebuilding")
    rows = []
    for f in sorted(SSL_DIR.glob("*.pt")):
        d = torch.load(f, map_location="cpu", mmap=True, weights_only=False)
        for uid, blk in d["series"].items():
            rows.append({"study": d["study"], "fold": int(d["fold"]), "uid": uid,
                         "n_planes": int(blk["planes"].shape[0])})
    df = pd.DataFrame(rows); df.to_parquet(INDEX, index=False)
    return df


class SSLPlanes(Dataset):
    """One item = one 2.5D triplet (c-1, c, c+1) from one series, clamped at the ends."""

    def __init__(self, index: pd.DataFrame, img: int = 336, epoch_len: int = 100_000,
                 scale=(0.8, 1.0), seed: int = 0):
        assert (index.fold != 4).all(), "fold 4 is reserved from SSL"
        self.by_study = {s: g[["uid", "n_planes"]].values.tolist()
                         for s, g in index.groupby("study")}
        self.studies = sorted(self.by_study)
        self.files = {s: SSL_DIR / f"{s}.pt" for s in self.studies}
        self.img, self.epoch_len, self.scale = img, epoch_len, scale
        self.seed = seed
        self._open: dict[str, dict] = {}

    def __len__(self): return self.epoch_len

    def _study(self, s):
        if s not in self._open:
            if len(self._open) > 64: self._open.pop(next(iter(self._open)))
            self._open[s] = torch.load(self.files[s], map_location="cpu", mmap=True,
                                       weights_only=False)
        return self._open[s]

    def __getitem__(self, i):
        # DETERMINISTIC in (seed, i): the sample is a pure function of its index, so the data
        # stream is reproducible and RESUMABLE from an offset -- no worker RNG state to save.
        # The previous worker-local RNG ignored `i` entirely, which made resume impossible and
        # made every worker's stream depend on process start order.
        r = random.Random(self.seed * 1_000_003 + i)
        s = r.choice(self.studies)                         # uniform over STUDIES
        uid, n = r.choice(self.by_study[s])                # uniform over that study's SERIES
        c = r.randrange(int(n))                            # uniform over that series' PLANES
        planes = self._study(s)["series"][uid]["planes"]
        idx = [max(0, c - 1), c, min(int(n) - 1, c + 1)]   # clamp, as the _nb cache does
        x = torch.from_numpy(np.stack([planes[j].numpy() for j in idx])).float().div_(255.0)
        if self.scale is not None:                          # light crop only; NO flips
            sc = r.uniform(*self.scale)
            side = max(8, int(round(self.img * sc)))
            top = r.randrange(0, self.img - side + 1)
            left = r.randrange(0, self.img - side + 1)
            x = F.interpolate(x[:, top:top + side, left:left + side].unsqueeze(0),
                              size=(self.img, self.img), mode="bilinear",
                              align_corners=False).squeeze(0)
        return x.sub_(IMNET_MEAN).div_(IMNET_STD)


class MAE(nn.Module):
    """Encoder = the real DINOv2 ViT (visible tokens only). Decoder = small, discarded after."""

    def __init__(self, encoder, img: int = 336, patch: int = 14, in_ch: int = 3,
                 dec_dim: int = 256, dec_depth: int = 4, dec_heads: int = 8,
                 mask_ratio: float = 0.75, norm_pix: bool = True):
        super().__init__()
        self.enc, self.patch, self.in_ch = encoder, patch, in_ch
        self.grid = img // patch
        self.n_patch = self.grid ** 2
        self.mask_ratio, self.norm_pix = mask_ratio, norm_pix
        d = encoder.embed_dim
        self.dec_embed = nn.Linear(d, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos = nn.Parameter(torch.zeros(1, self.n_patch, dec_dim))
        self.dec_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(dec_dim, dec_heads, dec_dim * 4, dropout=0.0,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.dec_head = nn.Linear(dec_dim, patch * patch * in_ch)
        nn.init.normal_(self.mask_token, std=0.02); nn.init.normal_(self.dec_pos, std=0.02)

    def patchify(self, x):
        B, C, H, W = x.shape; p, g = self.patch, self.grid
        return (x.reshape(B, C, g, p, g, p).permute(0, 2, 4, 3, 5, 1)
                 .reshape(B, g * g, p * p * C))

    def forward(self, x):
        B = x.shape[0]
        tok = self.enc.patch_embed(x)                                  # (B, 576, 384)
        tok = tok + self.enc.pos_embed[:, self.enc.num_prefix_tokens:]  # patch pos only
        keep = int(round(self.n_patch * (1 - self.mask_ratio)))
        noise = torch.rand(B, self.n_patch, device=x.device)
        order = noise.argsort(1); restore = order.argsort(1)
        vis = order[:, :keep]
        tok_v = torch.gather(tok, 1, vis[..., None].expand(-1, -1, tok.shape[-1]))
        cls = self.enc.cls_token.expand(B, -1, -1) + self.enc.pos_embed[:, :1]
        h = torch.cat([cls, tok_v], 1)                                 # masked patches OMITTED
        for blk in self.enc.blocks: h = blk(h)
        h = self.enc.norm(h)

        z = self.dec_embed(h[:, self.enc.num_prefix_tokens:])
        pad = self.mask_token.expand(B, self.n_patch - keep, -1)
        z = torch.cat([z, pad], 1)
        z = torch.gather(z, 1, restore[..., None].expand(-1, -1, z.shape[-1])) + self.dec_pos
        for blk in self.dec_blocks: z = blk(z)
        pred = self.dec_head(self.dec_norm(z))                         # (B, 576, p*p*C)

        tgt = self.patchify(x)
        if self.norm_pix:
            mu = tgt.mean(-1, keepdim=True); var = tgt.var(-1, keepdim=True)
            tgt = (tgt - mu) / (var + 1e-6).sqrt()
        mask = torch.ones(B, self.n_patch, device=x.device)
        mask.scatter_(1, vis, 0.0)                                     # 1 = masked = scored
        loss = (((pred - tgt) ** 2).mean(-1) * mask).sum() / mask.sum().clamp_min(1.0)
        return loss, keep
