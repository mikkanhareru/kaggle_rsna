from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rsna.data.geometry import SIDE_DEPENDENT_LABELS
from rsna.data.preprocess import load_series_volume
from rsna.labels.schema import COLUMN_MAP

LABELS = list(COLUMN_MAP.values())
IMAGENET_MEAN, IMAGENET_STD = 0.485, 0.229      # grayscale collapse of the RGB stats
PLANES = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


def series_type_id(plane: str, fluid_sensitive: int) -> int:
    """6 types: {Sag, Cor, Ax} x {T1/PD, fluid-sensitive+fatsat}.
    Fluid_Sensitive and Fat_Suppression are identical on all 24,371 series,
    so they carry one bit between them, not two."""
    return PLANES[plane] * 2 + int(fluid_sensitive)


def _sample_indices(n: int, k: int, train: bool, rng: np.random.Generator):
    """k slice indices spanning the stack.

    Training draws one slice at random inside each of k equal bins (stratified,
    so coverage is guaranteed but the exact slices vary each epoch). Eval takes
    bin centres, which is deterministic and reproducible.
    """
    if n <= 0:
        return np.zeros(k, int)
    edges = np.linspace(0, n, k + 1)
    if train:
        lo, hi = edges[:-1], edges[1:]
        idx = (lo + rng.random(k) * (hi - lo)).astype(int)
    else:
        idx = ((edges[:-1] + edges[1:]) / 2).astype(int)
    return np.clip(idx, 0, n - 1)


def _stack_25d(vol: torch.Tensor, idx: np.ndarray, gap: int = 1) -> torch.Tensor:
    """(k, 3, H, W) from neighbouring slices.

    Channels are [i-gap, i, i+gap] -- through-plane context, which is what makes
    an ImageNet-pretrained 3-channel backbone useful here rather than repeating
    one slice three times. Clamped at the stack boundaries.
    """
    n = vol.shape[0]
    tri = np.stack([np.clip(idx - gap, 0, n - 1), idx,
                    np.clip(idx + gap, 0, n - 1)], axis=1)   # (k, 3)
    return vol[torch.from_numpy(tri.reshape(-1))].reshape(len(idx), 3, *vol.shape[1:])


class StudyDataset(Dataset):
    """Returns one study: images, per-series metadata, targets and weights."""

    def __init__(self, studies: list[str], labels: pd.DataFrame,
                 series_meta: pd.DataFrame, folds: pd.DataFrame,
                 cache_dir: Path, *, n_series: int = 6, n_slices: int = 24,
                 img_size: int = 256, train: bool = True,
                 target_prefix: str = "y_", transform=None, seed: int = 0):
        self.studies = list(studies)
        self.lab = labels
        self.folds = folds
        self.cache = Path(cache_dir)
        self.n_series, self.n_slices = n_series, n_slices
        self.img_size, self.train = img_size, train
        self.tp, self.transform = target_prefix, transform
        self.seed = seed
        m = series_meta[series_meta.error.isna()]
        self.by_study = {s: d for s, d in m.groupby("StudyInstanceUID")}

    def __len__(self) -> int:
        return len(self.studies)

    def _pick_series(self, d: pd.DataFrame, rng) -> pd.DataFrame:
        """Choose up to n_series, covering PLANES before contrasts.

        The previous version iterated `sorted(pools)`, which always starts at
        type 0 (Sagittal/T1). At n_series=3 that took Sag-T1, Sag-FS, Cor-T1 and
        reached Axial in 1.6% of studies against its 24.2% corpus share -- the
        axial view is where PF OA and Baker's cyst are most visible.
        """
        d = d.assign(_t=[series_type_id(p, f) for p, f in
                         zip(d.plane, d.Fluid_Sensitive)])
        pools = {t: g.index.tolist() for t, g in d.groupby("_t")}
        for v in pools.values():
            rng.shuffle(v)
        types = sorted(pools)
        order, seen_planes = [], set()

        # pass 1: one series per PLANE, so a small n_series still spans Sag/Cor/Ax
        for t in rng.permutation(types):
            plane = t // 2
            if plane not in seen_planes and pools[t] and len(order) < self.n_series:
                order.append(pools[t].pop())
                seen_planes.add(plane)

        # pass 2: fill remaining slots round-robin over the contrast types
        while len(order) < self.n_series and any(pools.values()):
            for t in types:
                if pools[t] and len(order) < self.n_series:
                    order.append(pools[t].pop())
        return d.loc[order]

    def __getitem__(self, i: int) -> dict:
        uid = self.studies[i]
        rng = np.random.default_rng(None if self.train else (self.seed, i))
        d = self._pick_series(self.by_study[uid], rng)

        imgs, types, positions = [], [], []
        for _, r in d.iterrows():
            vol = load_series_volume(self.cache / f"{r.SeriesInstanceUID}.pt")
            idx = _sample_indices(vol.shape[0], self.n_slices, self.train, rng)
            x = _stack_25d(vol, idx).float().div_(255.0)          # (k,3,384,384)
            # Resize BEFORE augmenting. Augmenting at 384 then downscaling pays
            # 2.25x the pixels for every op -- measured 35.2 -> 10.7 ms/series,
            # a 3.3x speedup, and the dataloader is what starves the GPU here.
            if x.shape[-1] != self.img_size:
                x = torch.nn.functional.interpolate(
                    x, size=(self.img_size, self.img_size),
                    mode="bilinear", align_corners=False)
            if self.transform is not None:
                x = self.transform(x, rng)      # ONE transform per series: a
                                                # per-slice transform would make
                                                # the 2.5D channels misalign
            imgs.append(x.sub_(IMAGENET_MEAN).div_(IMAGENET_STD))
            types.append(series_type_id(r.plane, r.Fluid_Sensitive))
            positions.append(idx / max(vol.shape[0] - 1, 1))       # normalised depth

        row = self.lab.loc[uid]
        y = torch.tensor([row[f"{self.tp}{l}"] for l in LABELS], dtype=torch.float32)
        w = torch.tensor([row[f"w_{l}"] for l in LABELS], dtype=torch.float32)

        # Laterality unresolved -> medial/lateral is unreliable for this study.
        # TRAINING ONLY: validation must score all 12, because the test set does.
        if self.train and bool(self.folds.loc[uid, "lat_ambiguous"]):
            for j, l in enumerate(LABELS):
                if l in SIDE_DEPENDENT_LABELS:
                    w[j] = 0.0

        return dict(uid=uid,
                    img=torch.stack(imgs),                                  # (S,k,3,H,W)
                    series_type=torch.tensor(types, dtype=torch.long),      # (S,)
                    slice_pos=torch.tensor(np.stack(positions), dtype=torch.float32),
                    y=y, w=w,
                    gold=torch.tensor([row.get(f"gold_{l}", float("nan"))
                                       for l in LABELS], dtype=torch.float32))


def collate_studies(batch: list[dict]) -> dict:
    """Pad the variable series count and hand back a mask.

    Studies carry 3-14 series; padding to the batch max keeps a dense tensor
    while series_mask tells the aggregator which entries are real.
    """
    S = max(b["img"].shape[0] for b in batch)
    B = len(batch)
    k, c, h, w = batch[0]["img"].shape[1:]
    img = torch.zeros(B, S, k, c, h, w)
    stype = torch.zeros(B, S, dtype=torch.long)
    spos = torch.zeros(B, S, k)
    smask = torch.zeros(B, S, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b["img"].shape[0]
        img[i, :n], stype[i, :n], spos[i, :n], smask[i, :n] = (
            b["img"], b["series_type"], b["slice_pos"], True)
    # A study with zero real series would mask every key in the study-level
    # attention; softmax over all -inf returns NaN, which reaches submission.csv
    # as a silent NaN row rather than an error. Cannot happen today, but a
    # malformed test study at inference is exactly where it would.
    assert smask.any(dim=1).all(), "study with zero series would NaN the attention"
    out = dict(uid=[b["uid"] for b in batch], img=img, series_type=stype,
               slice_pos=spos, series_mask=smask,
               y=torch.stack([b["y"] for b in batch]),
               w=torch.stack([b["w"] for b in batch]),
               gold=torch.stack([b["gold"] for b in batch]))
    # Auxiliary targets only when EVERY item carries them (SlotDataset does, the
    # series/maxspan datasets do not) -- never a partially filled batch.
    if all("y_aux" in b for b in batch):
        out["y_aux"] = torch.stack([b["y_aux"] for b in batch])
        out["w_aux"] = torch.stack([b["w_aux"] for b in batch])
    return out