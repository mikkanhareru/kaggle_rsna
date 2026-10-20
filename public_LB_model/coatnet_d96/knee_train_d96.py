# %% [code]
# %% [code]
#!/usr/bin/env python3
"""Knee MRI: training the twelve-finding model

This is the training half of the model behind the public 0.924 inference notebook. It reads the
precomputed slice stacks, trains a CoAtNet backbone with a per-finding attention pooling head, and
writes a checkpoint you can drop straight into that notebook.

WHAT YOU NEED ATTACHED

  1. The preprocessed corpus, both parts:
       kaggle.com/datasets/dreaddevelopment/knee-raptor-corpus          (3,200 studies)
       kaggle.com/datasets/dreaddevelopment/knee-raptor-corpus-ext      (1,207 studies)
     Every study is already reduced to a fixed 44 x 336 x 336 uint8 stack, so no DICOM reading
     happens here. The two parts concatenate in order.

  2. The competition data, for train.csv.

  3. Training labels, as a parquet with a StudyInstanceUID column and the twelve finding columns.
     THIS IS NOT PROVIDED, and it is the one thing you have to bring yourself. See below.

THE LABEL PROBLEM, WHICH IS THE REAL PROBLEM

The competition gives you 4,407 studies and structured labels for only 58 of them. Every other
study carries a free-text radiology report and nothing else. So before any of this trains, you need
to turn 4,349 reports into twelve numbers each.

The approach behind the published weights was to read each report with a language model and emit
twelve probabilities rather than twelve yes or no answers: a report that hedges, saying a tear is
suspected, becomes something near 0.8 rather than a 1. Soft targets are far more forgiving than
forcing every hedged sentence into a hard label, and the loss here expects them. The 58 studies
that come with real labels are held out and used only for validation, never trained on.

Point --labels at your own parquet built that way. The format is one row per study: a
StudyInstanceUID column plus the twelve finding columns, values between 0 and 1.

WHAT THE MODEL DOES

Three neighbouring slices are stacked into the three channels of one image, so the network sees a
little of what lies above and below the middle slice: most of the benefit of a 3D model at the cost
of a 2D one. Each of these three-slice windows goes through the backbone, and the windows are then
pooled by an attention layer that has separate weights for each of the twelve findings. That last
part matters more than anything else here. A cruciate tear may be visible on two slices while
osteoarthritis spreads across many, and one shared pooling weight forces those to compete; giving
each finding its own attention lets each draw on the slices that actually show it.

Training samples k windows per study at random and evaluates on k_eval windows spread evenly, so
each epoch sees a different view of the same study. Nothing else is augmented.

At the end it keeps the best epoch by validation macro-AUC, and also writes a checkpoint that
averages the weights of the best three epochs. Weight averaging costs nothing at inference, unlike
averaging predictions from three models, and it usually gives a small gain.

TYPICAL RUN

  python train_knee.py --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --epochs 16
      --bs 8 --k 12 --k_eval 24 --grad_ckpt --tag mymodel --labels /kaggle/input/YOURS/labels.parquet

About three hours on one 4090 for 16 epochs at 384. --grad_ckpt trades a little speed for a lot of
memory and is what makes bs 8 fit on a 24 GB card. Use --smoke for a fast wiring check.

MEMORY, MEASURED

The CoAtNet at 384 with --bs 8 and --grad_ckpt sits at about 23.7 GB, so it needs a 24 GB card.
On 16 GB use --bs 2 or 4. Separately from the GPU, this script keeps the best checkpoint plus the
top --topk epochs' weights in system RAM so it can average them at the end; that is about 87 MB a
copy for the small ViT and 293 MB for the CoAtNet, and the list fills over the first few epochs.
If you run out of host memory a few epochs in, set --topk 1 and lower --workers.

The checkpoint it writes is a dict with keys model, arch, res and lab, which is exactly what the
inference notebook expects.
"""
import os, sys, time, json, math, random, argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import timm

# --- ROI localizer (anatomical joint crop). Optional so the no-ROI path is untouched. ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from roi_localize import square_box as _roi_square_box, compartments as _roi_compartments
except Exception:
    _roi_square_box = _roi_compartments = None

HERE = os.path.dirname(os.path.abspath(__file__))
RSNA = os.path.dirname(HERE)

# ---------------------------------------------------------------------------
# Input discovery. On Kaggle the corpus arrives as two read-only datasets and the
# competition data as a third, so nothing lives beside this script. Everything in
# this block is discovery only - the training code further down is unchanged.
# ---------------------------------------------------------------------------
def _find(*names, root="/kaggle/input"):
    """First path under root whose basename matches one of names."""
    for d, _, fs in os.walk(root):
        for n in names:
            if n in fs:
                return os.path.join(d, n)
    return None


class _TwoPartVols:
    """Presents the two published corpus parts as one array of shape (4407, 44, 336, 336).

    Both parts stay memory-mapped and are never concatenated on disk: copying 22 GB would be
    pointless when every read is a single study. Row order is part 1 then part 2, matching the
    order the id files concatenate in. That ordering is the contract between volumes, masks and
    ids, so do not sort any of them independently.
    """
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.n_a = a.shape[0]
        self.shape = (a.shape[0] + b.shape[0],) + tuple(a.shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, row):
        return self.a[row] if row < self.n_a else self.b[row - self.n_a]


def _open_corpus():
    """Return (vols, masks), from a local single-file corpus or the two public parts."""
    # DENSE96 arm: BOTH vols and masks are overridable. The CLIP arm overrode only vols,
    # which was safe there because the masks were asserted bitwise identical. Here the d96
    # masks are (N, 96) against the production (N, 64) -- loading the wrong one would
    # mis-index the window sampler rather than raise.
    local_v = os.environ.get("RSNA_CORPUS") or os.path.join(HERE, "all_vols.npy")
    local_m = os.environ.get("RSNA_MASKS") or os.path.join(HERE, "all_masks.npy")
    if os.path.exists(local_v):
        v = np.load(local_v, mmap_mode="r"); m = np.load(local_m)
        if v.shape[:2] != m.shape[:2]:
            raise SystemExit(f"corpus/mask shape mismatch: vols {v.shape} masks {m.shape}")
        return (v, m)
    av, bv = _find("all_vols.npy"), _find("extra_vols.npy")
    am, bm = _find("all_masks.npy"), _find("extra_masks.npy")
    if not all((av, bv, am, bm)):
        raise SystemExit("Could not find the corpus. Attach both parts: "
                         "dreaddevelopment/knee-raptor-corpus and "
                         "dreaddevelopment/knee-raptor-corpus-ext")
    vols = _TwoPartVols(np.load(av, mmap_mode="r"), np.load(bv, mmap_mode="r"))
    masks = np.concatenate([np.load(am), np.load(bm)], axis=0)
    return vols, masks


def _open_ids():
    local = os.path.join(HERE, "all_ids.npy")
    if os.path.exists(local):
        return np.load(local, allow_pickle=True).astype(str)
    a, b = _find("all_ids.npy"), _find("extra_ids.npy")
    if not (a and b):
        raise SystemExit("Could not find all_ids.npy / extra_ids.npy - attach both corpus parts.")
    return np.concatenate([np.load(a, allow_pickle=True).astype(str),
                           np.load(b, allow_pickle=True).astype(str)])
LAB = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA",
       "Effusion","Synovitis","Baker's","Contusion","Fracture"]


_NO_LABELS = """
No training labels found, so there is nothing to train against.

The competition labels only 58 of the 4,407 studies. The other 4,349 carry a free-text
radiology report instead, so before this can train you have to turn those reports into
twelve probabilities per study and pass the result with --labels.

Expected format: a parquet with a StudyInstanceUID column plus the columns
  {cols}
with values between 0 and 1. Soft values work better than hard 0/1 here: the loss is built
for them, and hedged reports are common.

Everything else in this notebook is ready to run once that file exists.
"""


# ------------------------------- data ----------------------------------------
class StudyWindows(Dataset):
    """Per-study bag of 2.5D windows sampled from all_vols.npy (memmap).
    Each window = 3 physically-consecutive slices -> RGB, resized to `res`, in [0,1]
    (matches the SSL input pipeline: ToTensor, no ImageNet norm)."""
    def __init__(self, root, ids, id2row, labels, res, k, train, aug=True, norm="none",
                 roi=False, roi_mode="tight", roi_pad=0.06, roi_overlap=0.12,
                 roi_sbox=None, roi_cen=None, wmask=None):
        self.root = root
        self.ids = ids
        self.id2row = id2row
        self.labels = labels              # dict uid -> np.float32[12]
        self.wmask = wmask                # RANK pilot: dict uid -> float32[12] report-mentioned mask, or None
        self.res, self.k, self.train, self.aug = res, k, train, aug
        self.norm = norm                  # "none"=[0,1] (Raptor SSL); "imagenet"=DINOv2 stats
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.vols = None; self.masks = None
        # --- ROI anatomical joint-crop config ---
        self.roi = bool(roi)
        self.roi_mode, self.roi_pad, self.roi_overlap = roi_mode, roi_pad, roi_overlap
        self.roi_sbox = roi_sbox          # (N,D,4) int16 per-slice tissue bbox, or None
        self.roi_cen = roi_cen            # (N,D,2) int16 per-slice joint centroid, or None
        if self.roi and (_roi_square_box is None or roi_sbox is None):
            raise RuntimeError("roi=True but roi_localize or roi_boxes not available")

    def __len__(self): return len(self.ids)

    def _ensure(self):
        if self.vols is None:
            self.vols, self.masks = _open_corpus()   # (N,D,H,W) view, (N,D) u8

    def _centers(self, valid, count):
        # valid slice indices; window centers must have both neighbors valid & in-range
        lo, hi = int(valid.min()), int(valid.max())
        cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
        if not cs: cs = [max(1, min((lo + hi) // 2, self._D - 2))]
        if self.train:
            reps = count // len(cs) + 1
            pool = (cs * reps)
            random.shuffle(pool)
            return pool[:count]
        # eval: evenly spaced deterministic
        idx = np.linspace(0, len(cs) - 1, count).round().astype(int)
        return [cs[i] for i in idx]

    def _resize(self, tri):
        """tri: (3,h,w) float32 [0,1] -> (3,res,res) float32."""
        t = torch.from_numpy(np.ascontiguousarray(tri))
        if t.shape[-1] != self.res or t.shape[-2] != self.res:
            t = F.interpolate(t[None], size=(self.res, self.res), mode="bilinear",
                              align_corners=False)[0]
        return t.numpy()

    # ---- REG pilot (competition_details/preregistered/2026-09-25_R_regularization.md) ----
    @staticmethod
    def _aug_params():
        """One draw per training window: rotation deg, scale, shift x, shift y (fraction of side), gamma, contrast."""
        return (random.uniform(-5.0, 5.0), random.uniform(0.95, 1.05), random.uniform(-0.025, 0.025),
                random.uniform(-0.025, 0.025), random.uniform(0.9, 1.1), random.uniform(0.9, 1.1))

    def _augment(self, tri, p=None):
        """tri: (3,h,w) float32 [0,1] -> (3,res,res). ONE bilinear resample does the affine AND the resize;
        the 3 slices share every parameter; pixels from outside the image -> 0."""
        rot, s, tx, ty, gam, con = self._aug_params() if p is None else p
        a = math.radians(rot)
        theta = torch.tensor([[math.cos(a) / s, -math.sin(a) / s, 2 * tx],
                              [math.sin(a) / s, math.cos(a) / s, 2 * ty]], dtype=torch.float32)
        # 1-px replicate border so the identity transform reproduces F.interpolate's edge clamping
        # exactly; theta is rescaled into the padded frame. Beyond that pixel, outside -> 0.
        H, W = tri.shape[-2], tri.shape[-1]
        theta = theta * torch.tensor([[W / (W + 2)], [H / (H + 2)]], dtype=torch.float32)
        t = F.pad(torch.from_numpy(np.ascontiguousarray(tri))[None], (1, 1, 1, 1), mode="replicate")
        grid = F.affine_grid(theta[None], [1, 3, self.res, self.res], align_corners=False)
        t = F.grid_sample(t, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]
        t = t.clamp(0, 1) ** gam                      # gamma
        m = t.mean(); t = (t - m) * con + m           # contrast about the window mean
        return t.numpy()

    def __getitem__(self, i):
        self._ensure()
        uid = self.ids[i]; row = self.id2row[uid]
        self._D = self.vols.shape[1]
        m = self.masks[row]
        valid = np.where(m > 0)[0]
        if len(valid) < 3: valid = np.arange(min(3, self._D))
        # compartment mode emits 2 crops/center -> sample ceil(k/2) centers to keep #windows==k
        compart = self.roi and self.roi_mode == "compartment"
        n_centers = (self.k + 1) // 2 if compart else self.k
        cs = self._centers(valid, n_centers)
        vol = self.vols[row]  # (D,H,W) u8  (single study read)
        tiles = []            # list of (3,res,res) float32
        for c in cs:
            c = max(1, min(c, self._D - 2))
            tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0  # (3,H,W)
            H, W = tri.shape[-2], tri.shape[-1]
            if not self.roi:
                tiles.append(self._augment(tri) if (self.train and self.aug) else self._resize(tri))   # REG pilot
                continue
            # one joint box from the CENTER slice, applied to all 3 slices (keeps RGB registered)
            sq_mode = "tight" if compart else self.roi_mode
            sq = _roi_square_box(tuple(int(v) for v in self.roi_sbox[row, c]), W=W, H=H,
                                 pad=self.roi_pad, mode=sq_mode,
                                 centroid=tuple(float(v) for v in self.roi_cen[row, c]))
            if compart:
                left, right = _roi_compartments(sq, overlap=self.roi_overlap)
                for box in (left, right):
                    x0, y0, x1, y1 = box
                    tiles.append(self._resize(tri[:, y0:y1, x0:x1]))
            else:
                x0, y0, x1, y1 = sq
                tiles.append(self._resize(tri[:, y0:y1, x0:x1]))
        if len(tiles) > self.k:
            tiles = tiles[:self.k]
        wins = np.stack(tiles, 0)          # (K,3,res,res)
        x = torch.from_numpy(wins)
        if self.train and self.aug:
            # light medical-safe aug: NO flips (laterality is signal); mild intensity jitter
            g = 1.0 + (random.random() - 0.5) * 0.20
            x = (x * g).clamp(0, 1)
        if self.norm == "imagenet":        # each backbone at its correct input distribution
            x = (x - self._mean) / self._std
        y = torch.from_numpy(self.labels[uid])
        if self.wmask is not None:                     # RANK pilot: training items also carry the mask
            return x, y, torch.from_numpy(self.wmask[uid])
        return x, y


def collate(batch):
    xs = torch.stack([b[0] for b in batch])   # (B,K,3,res,res)
    ys = torch.stack([b[1] for b in batch])   # (B,12)
    if len(batch[0]) > 2:                     # RANK pilot: (B,12) report-mentioned mask
        return xs, ys, torch.stack([b[2] for b in batch])
    return xs, ys


# ---- RANK pilot (competition_details/preregistered/2026-09-25_R_rank.md) ----
def pairwise_rank_loss(logits, y, w, tau=1.0):
    """Copied from rsna/train/loop.py pairwise_rank_loss (GLOBAL normalisation, no trim); parity is
    asserted by the Block 2 self-test. Pairs (i, j) with y_i > y_j, weighted by (y_i - y_j) * w_i * w_j."""
    d = logits.unsqueeze(1) - logits.unsqueeze(0)            # (B,B,L) z_i - z_j
    gap = y.unsqueeze(1) - y.unsqueeze(0)                    # (B,B,L) y_i - y_j
    pw = torch.clamp(gap, min=0.0) * (w.unsqueeze(1) * w.unsqueeze(0))
    per = F.softplus(-d / tau)
    tot = pw.sum()
    return logits.sum() * 0.0 if tot <= 0 else (pw * per).sum() / tot


# ------------------------------- model ---------------------------------------
def build_backbone(arch="vit_small_patch16_224", pretrained=False, drop_path=0.0):
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if drop_path > 0:
        kw["drop_path_rate"] = drop_path          # REG pilot: timm stochastic depth, linear in depth
    if is_vit:
        kw.update(global_pool="token", dynamic_img_size=True)
    else:
        kw.update(global_pool="avg")
    return timm.create_model(arch, **kw)


def load_raptor(bb, ckpt_path):
    if ckpt_path in ("timm", "pretrained"):
        return "timm-pretrained"
    if ckpt_path in ("", "none", "None"):
        print("[raptor] RANDOM-INIT control (no SSL weights)", flush=True)
        return "random-init"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    st = ck["student"] if "student" in ck else ck
    bbst = {k[len("backbone."):]: v for k, v in st.items() if k.startswith("backbone.")}
    missing, unexpected = bb.load_state_dict(bbst, strict=False)
    ep = ck.get("epoch", "?")
    print(f"[raptor] loaded backbone from {os.path.basename(ckpt_path)} (ssl epoch {ep}) | "
          f"loaded {len(bbst)} tensors, missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
    return f"{os.path.basename(ckpt_path)}@ep{ep}"


class RaptorClassifier(nn.Module):
    """Raptor encoder + per-diagnosis attention-MIL head (12 findings)."""
    def __init__(self, backbone, F_dim=384, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))
        return f.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x):
        return self.head(self.encode(x))


# ------------------------------- train ---------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # Defaults to ImageNet-pretrained weights, which is what every released arm was trained from.
    # The previous default pointed at a self-supervised checkpoint that is not distributed here,
    # so following the example run above failed on a missing file.
    ap.add_argument("--ckpt", default="timm",
                    help="'timm' for ImageNet-pretrained weights, or a path to your own "
                         "self-supervised checkpoint")
    ap.add_argument("--arch", default="vit_small_patch16_224")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--k_eval", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--bb_lr", type=float, default=3e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--freeze_blocks", type=int, default=0)
    ap.add_argument("--rank_w", type=float, default=0.0, help="RANK pilot: lambda on the pairwise rank term (0 = off)")
    ap.add_argument("--rank_tau", type=float, default=1.0, help="RANK pilot: logit scale inside the pair softplus")
    ap.add_argument("--rank_mask", default=None, help="RANK pilot: parquet with w_<finding> columns; pairs only where w > 0")
    ap.add_argument("--drop_path", type=float, default=0.0,
                    help="REG pilot: timm drop_path_rate (0 = reference behaviour)")
    # Each kept epoch is a full copy of the weights in system RAM (~87 MB small ViT, ~293 MB
    # CoAtNet). Lower to 1 on a memory-tight machine; at 1 no averaged checkpoint is written.
    ap.add_argument("--topk", type=int, default=3,
                    help="top epochs kept for weight averaging (1 disables averaging)")
    ap.add_argument("--norm", default="none", choices=["none", "imagenet"])
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--labels", default=None,
                    help="parquet of training labels: StudyInstanceUID + the twelve finding "
                         "columns, values 0..1. Not provided with this notebook - see the header.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    # ---- CV fold hook ----
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--fold_file", default=None)
    # ---- anatomical ROI joint-crop (A/B lever). Default OFF -> identical to baseline. ----
    ap.add_argument("--roi", action="store_true")
    ap.add_argument("--roi_mode", default="compartment", choices=["tight", "safe", "compartment"])
    ap.add_argument("--roi_pad", type=float, default=0.06)
    ap.add_argument("--roi_overlap", type=float, default=0.12)
    ap.add_argument("--roi_boxes", default=os.path.join(HERE, "roi_boxes.npz"))
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.smoke:
        a.epochs, a.bs, a.k, a.k_eval, a.limit, a.workers = 2, 4, 4, 6, 40, 0
    print(f"device {dev} | res {a.res} | k {a.k}/{a.k_eval} | bs {a.bs} | tag {a.tag} | "
          f"roi={a.roi}({a.roi_mode})", flush=True)

    # ---- ids / labels ----
    ids = _open_ids()
    id2row = {u: i for i, u in enumerate(ids)}
    idset = set(ids)
    _tcsv = os.path.join(RSNA, "train.csv")
    if not os.path.exists(_tcsv):
        _tcsv = _find("train.csv")
    if not _tcsv:
        raise SystemExit("Could not find train.csv - attach the competition data.")
    tr = pd.read_csv(_tcsv); tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    gold_df = tr[tr[LAB].notna().all(axis=1)].copy().set_index("StudyInstanceUID")
    gold_ids = [u for u in gold_df.index if u in idset]
    _lab = a.labels or _find("labels_llm_soft.parquet")
    if not _lab or not os.path.exists(_lab):
        # Exit cleanly rather than as a failure: running this notebook as published, with no
        # labels attached, is the expected path and should read as an explanation, not a crash.
        print(_NO_LABELS.format(cols=", ".join(LAB)), flush=True)
        raise SystemExit(0)
    soft = pd.read_parquet(_lab)
    soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str); soft = soft.set_index("StudyInstanceUID")
    goldset = set(gold_ids)
    train_ids = [u for u in ids if u in soft.index and u not in goldset]
    # ---- CV fold hook: hold out fold `a.fold`, train on the rest ----
    oof_ids = []
    if a.folds > 0:
        assert 0 <= a.fold < a.folds, f"--fold must be in [0,{a.folds}) when --folds>0"
        fmap = json.load(open(a.fold_file))["folds"]
        held = set(u for u in train_ids if fmap.get(u, -1) == a.fold)
        oof_ids = [u for u in train_ids if u in held]
        train_ids = [u for u in train_ids if u not in held]
        print(f"[cv] fold {a.fold}/{a.folds}: train {len(train_ids)} | OOF held-out {len(oof_ids)} "
              f"| fold_file {os.path.basename(a.fold_file)}", flush=True)
    if a.limit: train_ids = train_ids[:a.limit]
    labels = {u: soft.loc[u, LAB].values.astype(np.float32) for u in train_ids}
    for u in oof_ids: labels[u] = soft.loc[u, LAB].values.astype(np.float32)
    for u in gold_ids: labels[u] = gold_df.loc[u, LAB].values.astype(np.float32)
    print(f"train {len(train_ids)} | gold-val {len(gold_ids)}"
          + (f" | oof {len(oof_ids)}" if oof_ids else ""), flush=True)

    prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)
    pw = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev)
    wmask = None
    if a.rank_w > 0:                                   # RANK pilot: report-mentioned mask of the TRAINING studies
        _m = pd.read_parquet(a.rank_mask)
        _m = _m if _m.index.name == "StudyInstanceUID" else _m.set_index("StudyInstanceUID")
        _m.index = _m.index.astype(str)
        wmask = {u: (_m.loc[u, [f"w_{l}" for l in LAB]].values > 0).astype(np.float32) for u in train_ids}
        print(f"[rank] lambda {a.rank_w} tau {a.rank_tau} | mentioned share "
              f"{np.mean([m.mean() for m in wmask.values()]):.3f}", flush=True)

    # ---- ROI boxes (only when --roi) ----
    roi_sbox = roi_cen = None
    if a.roi:
        rb = np.load(a.roi_boxes)
        rb_ids = rb["ids"].astype(str)
        if not np.array_equal(rb_ids, ids):
            rmap = {u: i for i, u in enumerate(rb_ids)}
            missing = [u for u in ids if u not in rmap]
            if missing:
                raise RuntimeError(f"roi_boxes missing {len(missing)} corpus ids (e.g. {missing[:2]})")
            order = np.array([rmap[u] for u in ids])
            roi_sbox = rb["sbox"][order]; roi_cen = rb["cen"][order]
        else:
            roi_sbox = rb["sbox"]; roi_cen = rb["cen"]
        print(f"[roi] ENABLED mode={a.roi_mode} pad={a.roi_pad} overlap={a.roi_overlap} "
              f"boxes={os.path.basename(a.roi_boxes)} sbox={roi_sbox.shape}", flush=True)
    _roi_kw = dict(roi=a.roi, roi_mode=a.roi_mode, roi_pad=a.roi_pad, roi_overlap=a.roi_overlap,
                   roi_sbox=roi_sbox, roi_cen=roi_cen)

    tds = StudyWindows(HERE, train_ids, id2row, labels, a.res, a.k, train=True, norm=a.norm, wmask=wmask, **_roi_kw)
    vds = StudyWindows(HERE, gold_ids, id2row, labels, a.res, a.k_eval, train=False, norm=a.norm, **_roi_kw)
    tl = DataLoader(tds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                    collate_fn=collate, pin_memory=True, persistent_workers=a.workers > 0)
    vl = DataLoader(vds, batch_size=max(2, a.bs // 2), shuffle=False, num_workers=a.workers,
                    collate_fn=collate, persistent_workers=a.workers > 0)

    use_timm = a.ckpt in ("timm", "pretrained")
    bb = build_backbone(a.arch, pretrained=use_timm, drop_path=a.drop_path)
    src = load_raptor(bb, a.ckpt)
    if use_timm: print(f"[raptor] timm-pretrained backbone: {a.arch}", flush=True)
    F_dim = bb.num_features
    if a.grad_ckpt:
        try:
            bb.set_grad_checkpointing(True); print("[raptor] gradient checkpointing ON", flush=True)
        except Exception as e:
            print(f"[raptor] grad_ckpt unsupported for {a.arch}: {e}", flush=True)
    model = RaptorClassifier(bb, F_dim=F_dim).to(dev)
    if a.freeze_blocks > 0:
        # Backbones name their layer groups differently: ViT-style models use blocks.N., while
        # CoAtNet, MaxViT and ConvNeXt use stages.N. Matching only one of those froze nothing at
        # all on the other family, and did it silently. Try both, and always say what happened.
        pref = tuple([f"blocks.{b}." for b in range(a.freeze_blocks)]
                     + [f"stages.{b}." for b in range(a.freeze_blocks)])
        frozen = 0
        for nm, p in model.backbone.named_parameters():
            if nm.startswith(pref):
                p.requires_grad = False
                frozen += p.numel()
        n_t = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
        if frozen:
            print(f"[raptor] froze the first {a.freeze_blocks} layer group(s): "
                  f"{n_t} tensors, {frozen/1e6:.2f}M parameters", flush=True)
        else:
            print(f"[raptor] WARNING: --freeze_blocks {a.freeze_blocks} matched no parameters on "
                  f"{a.arch}; every parameter is still being trained", flush=True)

    head_params = [p for n_, p in model.named_parameters() if not n_.startswith("backbone.") and p.requires_grad]
    bb_params = [p for n_, p in model.named_parameters() if n_.startswith("backbone.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb_params, "lr": a.bb_lr},
                             {"params": head_params, "lr": a.head_lr}], weight_decay=a.wd)
    steps = max(len(tl) * a.epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr],
                                                total_steps=steps, pct_start=0.15)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    @torch.no_grad()
    def evaluate():
        model.eval(); P = []; Y = []
        for x, y in vl:
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                o = torch.sigmoid(model(x.to(dev)).float())
            P.append(o.cpu().numpy()); Y.append(y.numpy())
        P = np.concatenate(P); Y = np.concatenate(Y)
        aucs = {}
        for j, name in enumerate(LAB):
            if len(set(Y[:, j].astype(int))) > 1:
                aucs[name] = float(roc_auc_score(Y[:, j], P[:, j]))
        return float(np.mean(list(aucs.values()))), aucs, P, Y

    best = 0.0; best_state = None; best_P = None; t0 = time.time(); hist = []
    TOPK = max(1, a.topk); topk = []
    for ep in range(a.epochs):
        model.train(); tot = 0.0; rk_sum = 0.0; bce_sum = 0.0; rk_zero = 0     # RANK pilot diagnostics
        for batch in tl:
            x, y = batch[0].to(dev, non_blocking=True), batch[1].to(dev, non_blocking=True)
            opt.zero_grad()
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(x)
                loss = lossf(logits.float(), y)
                if a.rank_w > 0:                       # RANK pilot: + lambda * rank over mentioned pairs
                    rk = pairwise_rank_loss(logits.float(), y, batch[2].to(dev, non_blocking=True), a.rank_tau)
                    bce_sum += loss.item(); rk_sum += rk.item(); rk_zero += int(rk.item() == 0.0)
                    loss = loss + a.rank_w * rk
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step(); sched.step(); tot += loss.item()
        au, aucs, P, Y = evaluate()
        hist.append({"ep": ep, "loss": tot / len(tl), "gold_auc": au})
        if au > best:
            best = au; best_P = P
            best_state = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          "gold_auc": au, "aucs": aucs, "src": src, "res": a.res,
                          "arch": a.arch, "lab": LAB, "epoch": ep}
        if au >= best and best_state is not None:
            _tmp = os.path.join(HERE, f"raptor_ft_{a.tag}.pt.tmp")
            torch.save(best_state, _tmp)
            os.replace(_tmp, os.path.join(HERE, f"raptor_ft_{a.tag}.pt"))
            np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}.npz"),
                     pred=best_P, truth=Y, ids=np.array(gold_ids))
            json.dump({"tag": a.tag, "src": src, "best_gold_auc": best,
                       "aucs": best_state["aucs"], "hist": hist, "res": a.res,
                       "epochs": a.epochs, "bb_lr": a.bb_lr, "head_lr": a.head_lr,
                       "n_train": len(train_ids), "n_gold": len(gold_ids),
                       "roi": a.roi, "roi_mode": a.roi_mode,
                       "partial": True, "epochs_done": ep + 1},
                      open(os.path.join(HERE, f"raptor_ft_{a.tag}.json"), "w"), indent=1)
            print(f"  [ckpt] best-so-far saved at ep{ep} ({best:.4f})", flush=True)
        if len(topk) < TOPK or au > min(t["gold_auc"] for t in topk):
            topk.append({"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                         "gold_auc": au, "aucs": aucs, "src": src, "res": a.res,
                         "arch": a.arch, "lab": LAB, "epoch": ep, "P": P})
            topk.sort(key=lambda t: -t["gold_auc"])
            del topk[TOPK:]
        print(f"ep{ep} loss {tot/len(tl):.3f} | GOLD macro-AUC {au:.4f} (best {best:.4f}) | {time.time()-t0:.0f}s",
              flush=True)
        if a.rank_w > 0:                               # RANK pilot diagnostics (separate line; the one above is parsed)
            print(f"  [rank] ep{ep} bce {bce_sum/len(tl):.4f} | rank {rk_sum/len(tl):.4f} | zero-pair steps "
                  f"{rk_zero/len(tl):.1%}", flush=True)
    _, aucs, _, Y = evaluate()
    print(f"\nDONE {src} | BEST GOLD macro-AUC {best:.4f}", flush=True)
    for k, v in (best_state["aucs"] if best_state else aucs).items():
        print(f"   {k:18s} {v:.3f}", flush=True)

    if best_state is not None:
        torch.save(best_state, os.path.join(HERE, f"raptor_ft_{a.tag}.pt"))
        np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}.npz"),
                 pred=best_P, truth=Y, ids=np.array(gold_ids))
        json.dump({"tag": a.tag, "src": src, "best_gold_auc": best, "aucs": best_state["aucs"],
                   "hist": hist, "res": a.res, "epochs": a.epochs, "bb_lr": a.bb_lr,
                   "head_lr": a.head_lr, "n_train": len(train_ids), "n_gold": len(gold_ids),
                   "roi": a.roi, "roi_mode": a.roi_mode, "partial": False, "epochs_done": a.epochs},
                  open(os.path.join(HERE, f"raptor_ft_{a.tag}.json"), "w"), indent=1)
        print(f"saved raptor_ft_{a.tag}.pt / raptor_gold_{a.tag}.npz / raptor_ft_{a.tag}.json", flush=True)

    # ---- CV OOF: predict the held-out fold at the best (gold-selected) weights ----
    # PILOT DIAGNOSTIC (competition_details/preregistered/2026-09-25_R_regularization.md): capture the
    # TERMINAL weights BEFORE the OOF block reloads best_state. This runs AFTER the training loop has
    # finished, so it cannot alter training, ordering or RNG consumption.
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "arch": a.arch, "res": a.res, "lab": LAB, "epoch": "terminal", "src": src},
               os.path.join(HERE, f"raptor_ft_{a.tag}_terminal.pt"))
    print(f"saved raptor_ft_{a.tag}_terminal.pt (terminal weights, diagnostic only)", flush=True)
    if a.folds > 0 and oof_ids and best_state is not None:
        model.load_state_dict(best_state["model"]); model.eval()
        ods = StudyWindows(HERE, oof_ids, id2row, labels, a.res, a.k_eval, train=False, norm=a.norm, **_roi_kw)
        ol = DataLoader(ods, batch_size=max(2, a.bs // 2), shuffle=False, num_workers=a.workers,
                        collate_fn=collate, persistent_workers=False)
        Po = []
        with torch.no_grad():
            for x, y in ol:
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                    o = torch.sigmoid(model(x.to(dev)).float())
                Po.append(o.cpu().numpy())
        Po = np.concatenate(Po)
        Yo = np.stack([labels[u] for u in oof_ids]).astype(np.float32)
        np.savez(os.path.join(HERE, f"raptor_oof_{a.tag}_fold{a.fold}.npz"),
                 pred=Po, truth=Yo, ids=np.array(oof_ids), fold=a.fold, nfolds=a.folds)
        print(f"[cv] wrote raptor_oof_{a.tag}_fold{a.fold}.npz ({len(oof_ids)} studies; "
              f"truth = soft labels)", flush=True)

        # --- top-K epoch OOF (new) ---
        # The epoch-ensemble used to be judged on gold only; that gate is too small to
        # resolve the move. Re-run the held-out fold at each retained epoch instead.
        oof_by_ep = {best_state["epoch"]: Po}
        for t in topk:
            e = int(t["epoch"])
            if e in oof_by_ep:
                continue
            model.load_state_dict(t["model"]); model.eval()
            Pe = []
            with torch.no_grad():
                for x, y in ol:
                    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                        Pe.append(torch.sigmoid(model(x.to(dev)).float()).cpu().numpy())
            Pe = np.concatenate(Pe); oof_by_ep[e] = Pe
            np.savez(os.path.join(HERE, f"raptor_oof_{a.tag}_ep{e}_fold{a.fold}.npz"),
                     pred=Pe, truth=Yo, ids=np.array(oof_ids), fold=a.fold, nfolds=a.folds)
            print(f"[cv] wrote top-K epoch OOF ep{e}", flush=True)
        if len(oof_by_ep) > 1:
            Pens = np.mean(list(oof_by_ep.values()), axis=0)
            np.savez(os.path.join(HERE, f"raptor_oof_{a.tag}_epens_fold{a.fold}.npz"),
                     pred=Pens, truth=Yo, ids=np.array(oof_ids), fold=a.fold, nfolds=a.folds)
            print(f"[cv] wrote epoch-ensemble OOF over epochs {sorted(oof_by_ep)}", flush=True)
        model.load_state_dict(best_state["model"]); model.eval()

    # --- weight-averaged checkpoint (new) ---
    if len(topk) > 1:
        import copy
        sds = [t["model"] for t in topk]
        avg = {}
        for k in sds[0]:
            v0 = sds[0][k]
            if v0.is_floating_point():
                avg[k] = sum(sd[k].double() for sd in sds).div(len(sds)).to(v0.dtype)
            else:
                avg[k] = v0.clone()          # e.g. num_batches_tracked
        swa_state = {"model": avg, "gold_auc": None, "aucs": {}, "src": src, "res": a.res,
                     "arch": a.arch, "lab": LAB, "epoch": [int(t["epoch"]) for t in topk],
                     "swa_over": [int(t["epoch"]) for t in topk]}
        model.load_state_dict(avg); model.eval()
        au_swa, aucs_swa, P_swa, Y_swa = evaluate()
        swa_state["gold_auc"] = au_swa; swa_state["aucs"] = aucs_swa
        torch.save(swa_state, os.path.join(HERE, f"raptor_ft_{a.tag}_swa.pt"))
        np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}_swa.npz"),
                 pred=P_swa, truth=Y_swa, ids=np.array(gold_ids))
        print(f"SWA over epochs {[int(t['epoch']) for t in topk]} | gold {au_swa:.4f} "
              f"(best-epoch {best:.4f}) {'BETTER' if au_swa > best else 'no gain'}", flush=True)
        if a.folds > 0 and oof_ids:
            ods2 = StudyWindows(HERE, oof_ids, id2row, labels, a.res, a.k_eval, train=False,
                                norm=a.norm, **_roi_kw)
            ol2 = DataLoader(ods2, batch_size=max(2, a.bs // 2), shuffle=False,
                             num_workers=a.workers, collate_fn=collate, persistent_workers=False)
            Ps = []
            with torch.no_grad():
                for x, y in ol2:
                    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                        Ps.append(torch.sigmoid(model(x.to(dev)).float()).cpu().numpy())
            Ps = np.concatenate(Ps)
            np.savez(os.path.join(HERE, f"raptor_oof_{a.tag}_swa_fold{a.fold}.npz"),
                     pred=Ps, truth=np.stack([labels[u] for u in oof_ids]).astype(np.float32),
                     ids=np.array(oof_ids), fold=a.fold, nfolds=a.folds)
            print(f"[cv] wrote SWA OOF", flush=True)
        model.load_state_dict(best_state["model"]); model.eval()

    if len(topk) > 1:
        eps = [t["epoch"] for t in topk]
        for rank, t in enumerate(topk):
            if rank == 0:
                continue
            P_t = t.pop("P")
            np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}_ep{t['epoch']}.npz"),
                     pred=P_t, truth=Y, ids=np.array(gold_ids))
            t["P"] = P_t
        Pens = np.mean([t["P"] for t in topk], axis=0)
        ens_aucs = {}
        for j, name in enumerate(LAB):
            if len(set(Y[:, j].astype(int))) > 1:
                ens_aucs[name] = float(roc_auc_score(Y[:, j], Pens[:, j]))
        ens = float(np.mean(list(ens_aucs.values())))
        np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}_epens.npz"),
                 pred=Pens, truth=Y, ids=np.array(gold_ids))
        print(f"TOPK epochs {eps} | best {best:.4f} | epoch-ensemble {ens:.4f} "
              f"({'BETTER' if ens > best else 'no gain'})", flush=True)


if __name__ == "__main__":
    main()