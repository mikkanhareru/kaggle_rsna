"""MaxSpan representation -- a VERBATIM reproduction of the public 0.924 notebook.

Deliberately NOT a better version of our slot cache. Its whole value is being a
SECOND VIEW whose errors are uncorrelated with ours, so every place where their
choice looks wrong is preserved on purpose:

  their choice                        our slot cache does           kept because
  ------------------------------------------------------------------------------
  first matching series               longest / best quality        divergence is
  trusts provided Fluid_Sensitive     recovers it from headers      the POINT --
  no laterality canonicalisation      mirrors L knees to R          a "fixed"
  centre crop, no content bbox        content-aware crop            reproduction
  windows straddle slot boundaries    slots stay separate           would no
  2-98% span, 64 slices               20-80% span, 12/slot          longer be a
                                                                    second view

Only once the checkpoint reproduces its stored per-label AUCs on gold58 do we
know the reproduction is faithful. Improvements go in a SEPARATE module.

Layout of the 64-slice stack: slot identity is carried by POSITION alone. A
missing slot leaves its k slices zero and still advances the index, so slice 0-17
is always sagittal-fluid, 18-31 always sagittal-other, and so on.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

IMG = 336
CROP_MM = 140.0
SPAN = (0.02, 0.98)
PCTL = (2.0, 98.0)
SLOTS = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
         ("Coronal", 0, 8), ("Axial", -1, 12)]
MAXS = sum(s[2] for s in SLOTS)          # 64
# 24, NOT the notebook's 62. The checkpoint stores the per-label gold58 AUCs it
# was validated at, and only k=24 reproduces them -- all twelve to 4 decimals
# (span 0.02-0.98). The shipped notebook evaluates at 62 and its prose claims 42;
# on gold58 those score 0.9198 and 0.9169 against 0.9213 at 24, so the released
# inference path is a regression against the setting the weights were checked at.
K_EVAL = 24
RES = 384
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
       "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
       "Contusion", "Fracture"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)

# Slot boundaries in the 64-slice stack, for OUR variant later. Unused by the
# verbatim path -- recorded here so the two cannot drift.
SLOT_BOUNDS = np.cumsum([0] + [s[2] for s in SLOTS])


def order_and_meta(sdir: str):
    """Sort a series by physical position. Verbatim: dot(IPP, slice_normal)."""
    import pydicom
    fs = glob.glob(sdir + "/*.dcm")
    recs, ps_list = [], []
    for f in fs:
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            iop = getattr(h, "ImageOrientationPatient", None)
            ipp = getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                r = np.array(iop[:3], float)
                c = np.array(iop[3:], float)
                pos = float(np.dot(np.array(ipp, float), np.cross(r, c)))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            ps = getattr(h, "PixelSpacing", None)
            ps = float(ps[0]) if ps is not None else 0.5
            ps_list.append(ps)
            recs.append((pos, f, ps))
        except Exception:
            recs.append((0.0, f, 0.5))
    recs.sort(key=lambda x: x[0])
    med = float(np.median(ps_list)) if ps_list else 0.5
    return [(f, ps) for _, f, ps in recs], med


def read_px(f: str) -> np.ndarray:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut
    d = pydicom.dcmread(f)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def mm_crop_resize(a: np.ndarray, ps: float) -> np.ndarray:
    """Centre crop CROP_MM using PixelSpacing[0] for BOTH axes, then resize.

    Note the failure mode we keep: `min(cpx, min(h,w))` means a scan whose FOV is
    smaller than 140 mm silently yields a crop that is NOT 140 mm. The physical
    scale guarantee is therefore only one-sided.
    """
    import cv2
    h, w = a.shape
    cpx = int(round(CROP_MM / max(ps, 1e-3)))
    cpx = min(cpx, min(h, w))
    y0, x0 = (h - cpx) // 2, (w - cpx) // 2
    a = a[y0:y0 + cpx, x0:x0 + cpx]
    return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)


def pick_series(rows, plane, fluid, used):
    """First matching series, not the longest. `used` forces distinct series.

    When a study has only one sagittal series, slot 1 takes it and slot 2 falls
    through to `cands[0]` of what remains -- so the "non fluid-sensitive" slot is
    really just "a second series in this plane", whatever its flag says.
    """
    cands = [r for r in rows
             if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_study(sid: str, rows: list[dict], series_root: str):
    """-> (vol uint8 (64,336,336), mask uint8 (64,)). Verbatim `build_study`."""
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    idx, used = 0, set()
    for plane, fluid, k in SLOTS:
        r = pick_series(rows, plane, fluid, used)
        if r is None:
            idx += k
            continue
        used.add(r["SeriesInstanceUID"])
        files, med_ps = order_and_meta(f"{series_root}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            idx += k
            continue
        n = len(files)
        lo, hi = int(n * SPAN[0]), int(n * SPAN[1]) - 1
        hi = max(hi, lo)
        picks = (np.linspace(lo, hi, k).round().astype(int) if n > 1
                 else np.zeros(k, int))
        arrs, pss = [], []
        for p in picks:
            fp, ps = files[min(int(p), n - 1)]
            try:
                arrs.append(read_px(fp)); pss.append(ps)
            except Exception:
                arrs.append(None); pss.append(med_ps)
        # Window over the SELECTED slices only -- not the whole series. Slices we
        # never look at cannot drag the percentiles.
        valid = [a for a in arrs if a is not None]
        if valid:
            loq, hiq = np.percentile(np.concatenate([a.ravel() for a in valid]), PCTL)
        else:
            loq, hiq = 0.0, 1.0
        for a, ps in zip(arrs, pss):
            if idx >= MAXS:
                break
            if a is None:
                idx += 1
                continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)
            vol[idx] = (aw * 255).astype(np.uint8)
            idx += 1
        if idx >= MAXS:
            break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


# ---------------------------------------------------------------- windowing --
def eval_centers(mask: np.ndarray, D: int, k: int) -> list[int]:
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]


def eval_windows(vol: np.ndarray, mask: np.ndarray, k: int = K_EVAL, res: int = RES):
    """-> torch (k,3,res,res), imagenet-normalised. Windows may straddle slots."""
    import torch
    import torch.nn.functional as F
    D = vol.shape[0]
    out = np.empty((k, 3, res, res), np.float32)
    for j, c in enumerate(eval_centers(mask, D, k)):
        c = max(1, min(c, D - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        if tri.shape[-1] != res:
            t = torch.from_numpy(tri)[None]
            tri = F.interpolate(t, size=(res, res), mode="bilinear",
                                align_corners=False)[0].numpy()
        out[j] = tri
    return torch.from_numpy((out - MEAN) / STD)


# ------------------------------------------------------------------- model ---
def build_backbone(arch: str, pretrained: bool = False):
    import timm
    # "coatnet"/"maxvit" contain no ViT token stream: the substring "vit" must not
    # route them down the token-pool path.
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in
                                  ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    kw.update(dict(global_pool="token", dynamic_img_size=True) if is_vit
              else dict(global_pool="avg"))
    return timm.create_model(arch, **kw)


def make_classifier(F_dim: int, n: int = 12, drop: float = 0.2):
    import torch
    import torch.nn as nn

    class RaptorClassifier(nn.Module):
        """Per-label attention over windows: `att` emits n scores per window and
        softmaxes over WINDOWS, so each finding pools its own slice subset. The
        head is a per-label dot product, not a shared Linear."""

        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.norm = nn.LayerNorm(F_dim)
            self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(),
                                     nn.Dropout(drop), nn.Linear(256, n))
            self.clsW = nn.Parameter(torch.zeros(n, F_dim))
            self.clsb = nn.Parameter(torch.zeros(n))
            nn.init.trunc_normal_(self.clsW, std=0.02)

        def encode(self, x):
            B, K = x.shape[:2]
            return self.backbone(x.flatten(0, 1)).view(B, K, -1)

        def head(self, feats):
            h = self.norm(feats)
            a = torch.softmax(self.att(h), dim=1)
            pooled = torch.einsum("bkn,bkf->bnf", a, h)
            return (pooled * self.clsW).sum(-1) + self.clsb

        def forward(self, x):
            return self.head(self.encode(x))

    return RaptorClassifier


def load_model(pt_path, device="cuda"):
    import torch
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    bb = build_backbone(ck["arch"], pretrained=False)
    model = make_classifier(bb.num_features)(bb)
    model.load_state_dict(ck["model"], strict=True)   # strict: a silent partial
    model.eval().to(device)                            # load would fake a result
    meta = {k: v for k, v in ck.items() if k != "model"}
    del ck
    return model, meta
