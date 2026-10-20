# DICOM series -> canonical uint8 volume. Used by BOTH the training cache
# builder and the Kaggle inference notebook, so the two cannot drift apart.

from __future__ import annotations

import os 
from collections import Counter
from pathlib import Path

import torch
import cv2 
import numpy as np 
import pydicom

from rsna.data.geometry import (AMBIGUOUS_MM, SliceHeader, series_geometry,
                                slice_center)

HEADER_TAGS = ["ImagePositionPatient", "ImageOrientationPatient", "Rows",
               "Columns", "PixelSpacing", "Laterality"]
PCT_LO, PCT_HI = 0.5, 99.5      # per-series robust window; MRI has no absolute scale
FG_THRESH = 0.08                # foreground cut for the crop box
MIN_CROP_FRAC = 0.25            # refuse a crop that keeps less than this
CACHE_SUFFIX = ".pt"

def read_series_headers(series_dir: Path):
    """(headers, paths, laterality_tag). Pixels are not touched."""
    heads, paths, tag = [], [], None
    for f in sorted(os.listdir(series_dir)):
        p = series_dir / f
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, specific_tags=HEADER_TAGS)
            ps = [float(x) for x in ds.PixelSpacing]
            heads.append(SliceHeader(
                ipp=np.array(ds.ImagePositionPatient, float),
                iop=np.array(ds.ImageOrientationPatient, float),
                rows=int(ds.Rows), cols=int(ds.Columns),
                pixel_spacing=(ps[0], ps[1])))
            paths.append(p)
            tag = tag or getattr(ds, "Laterality", None)
        except Exception:
            pass
    return heads, paths, tag

def resolve_study_laterality(per_series: list[tuple[float, str | None]]):
    """One laterality for the whole study -> (laterality, source).

    Per-series resolution is equally accurate (96.0% vs 95.9%) but leaves 1.2%
    of studies whose series disagree, which would mirror a study's coronal and
    axial views in OPPOSITE directions and feed the study-level aggregator
    contradictory evidence about which side is medial.
    """
    tags = [t for _, t in per_series if t in ("L", "R")]
    if tags:
        return Counter(tags).most_common(1)[0][0], "tag"
    votes = ["L" if cx > 0 else "R" for cx, _ in per_series]
    lat = Counter(votes).most_common(1)[0][0]
    if all(abs(cx) < AMBIGUOUS_MM for cx, _ in per_series):
        return lat, "ambiguous"          # caller should mask SIDE_DEPENDENT_LABELS
    return lat, "geometry"

def _read_pixels(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(path)
    a = ds.pixel_array.astype(np.float32)
    a = a * float(getattr(ds, "RescaleSlope", 1)) + float(getattr(ds, "RescaleIntercept", 0))
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        a = a.max() - a             # invert so bright always means high signal
    return a

def _crop_box(vol: np.ndarray) -> tuple[int, int, int, int]:
    """One box for the WHOLE series.

    Deliberately not per-slice: a per-slice box makes the anatomy jitter between
    adjacent slices, which breaks the 2.5D three-channel stacking downstream.
    """
    proj = vol.max(axis=0)
    ys, xs = np.where(proj > FG_THRESH)
    if ys.size == 0:
        return 0, vol.shape[1], 0, vol.shape[2]
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    if (y1 - y0) < MIN_CROP_FRAC * vol.shape[1] or (x1 - x0) < MIN_CROP_FRAC * vol.shape[2]:
        return 0, vol.shape[1], 0, vol.shape[2]      # implausible, keep everything
    return int(y0), int(y1), int(x0), int(x1)

def _fit_square(img: np.ndarray, size: int) -> np.ndarray:
    """Resize the long side to `size`, then centre-pad to square.
    Padding rather than stretching -- anatomy must keep its aspect ratio."""
    h, w = img.shape
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size), np.float32)
    y, x = (size - nh) // 2, (size - nw) // 2
    out[y:y + nh, x:x + nw] = r
    return out

def build_series_volume(series_dir: Path, size: int = 384,
                        laterality: str | None = None,
                        min_valid_frac: float = 0.5):
    """Canonical uint8 volume (N, size, size) plus metadata.

    A single unreadable slice no longer costs the whole series: bad slices are
    dropped, counted, and the stack is marked irregular. The series is failed
    only if fewer than `min_valid_frac` of its slices survive.
    """
    heads, paths, tag = read_series_headers(series_dir)
    if not heads:
        raise ValueError(f"no readable headers in {series_dir}")
    g = series_geometry(heads, laterality_tag=laterality or tag)

    keep, n_corrupt = [], 0
    for i in g.order:                       # canonical order
        try:
            keep.append(_read_pixels(paths[i]))
        except Exception:
            n_corrupt += 1
    if len(keep) < max(2, min_valid_frac * len(g.order)):
        raise ValueError(f"only {len(keep)}/{len(g.order)} slices readable")
    if len({a.shape for a in keep}) > 1:
        raise ValueError("inconsistent slice shapes within series")

    vol = np.stack(keep)
    lo, hi = np.percentile(vol, [PCT_LO, PCT_HI])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    y0, y1, x0, x1 = _crop_box(vol)
    vol = np.stack([_fit_square(s, size) for s in vol[:, y0:y1, x0:x1]])
    if g.flip_lr:
        vol = vol[:, :, ::-1]

    meta = dict(
        n_slices=len(vol), n_corrupt=n_corrupt, plane=g.plane,
        laterality=g.laterality, lat_source=g.lat_source, flip_lr=bool(g.flip_lr),
        spacing=g.spacing, irregular=bool(g.irregular or n_corrupt > 0),
        center_x=g.center_x, crop=(y0, y1, x0, x1),
        orig_hw=(heads[0].rows, heads[0].cols), size=size,
    )
    return np.ascontiguousarray((vol * 255).astype(np.uint8)), meta

def save_series_volume(path: Path, vol: np.ndarray) -> None:
    """uint8 (N, N, N) -> .pt."""
    tmp = path.with_suffix(".tmp")
    torch.save(torch.from_numpy(vol), tmp)
    tmp.replace(path)
    
def load_series_volume(path: Path, idx=None) -> torch.Tensor:
    """Read a cached series, optionally only the slices in 'idx'
    ALWAYS use this rather than a bare torch.load. mmap=True reads only the
    pages actually indexed: 0.71 ms vs 25.4 ms on a 320-slice series, because
    a plain load materialises every slice to use the ~24 we asked for.
    weights_only=True keeps it off the pickle path."""
    
    t = torch.load(path, weights_only=True, mmap=True)
    return t if idx is None else t[idx]

    
    
    