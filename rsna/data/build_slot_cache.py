"""Slot cache: study x 6 slots x 12 slices x 336 x 336, uint8.

A second, independent cache. It does NOT replace `cache/`, which backs the
verified 0.774 baseline and must stay reproducible.

Three things differ from `cache/`:

  1. A constant 130 mm crop instead of an intensity-threshold foreground box.
     The old box is content-driven, so physical scale varies 3.1x across series.
     A fixed extent over a fixed pixel count is a fixed pitch: 130/336 = 0.387
     mm/px, under the 0.5 mm Nyquist bound for a 1 mm meniscal tear. Native
     acquisition pitch is 0.312 mm/px median and 130 mm spans 416 px at the
     median, so 336 downsamples for 71% of series -- it resolves rather than
     interpolates. 336 is also forced by DINOv2 patch-14: 336/14 = 24 exactly,
     while 384/14 is not an integer.

  2. Six medically defined slots instead of six nominal series types. The
     provided Fluid_Sensitive and Fat_Suppression columns are identical on all
     24,371 rows -- one axis wearing two names. Weighting is recovered from the
     headers; fat suppression is taken from the provided column, which agrees
     with header recovery on 98.2% of series and, unlike a header parse, is
     equally available at inference.

  3. Twelve slices per slot over the central 20-80% of the ordered stack, read
     directly rather than decoded whole. Coverage is the cheap axis and
     resolution the expensive one -- the cache grows linearly in slices and
     quadratically in pixels.

Slice ordering, study-level laterality and the canonical mirror are reused from
`geometry.py` unchanged; they are more careful than the public baseline's, which
mirrors with a bare flip and ignores the row-direction sign.

    python -m rsna.data.audit_slots --split train --dump   # header table first
    python -m rsna.data.build_slot_cache --limit 40        # inspection batch
    python -m rsna.data.build_slot_cache                   # full build (130 mm)
    python -m rsna.data.build_slot_cache --crop-mm 160     # -> its OWN directory

`--crop-mm` trades pitch for coverage: 160/336 = 0.476 mm/px, still under the
0.5 mm bound. The output directory is DERIVED from the parameters, so a second
crop can never be written into or resumed against the first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from rsna.data import sequence
from rsna.data.geometry import series_geometry
from rsna.data.preprocess import (_read_pixels, read_series_headers,
                                  resolve_study_laterality)

ROOT = Path(__file__).resolve().parents[2]

# Default crop. `--crop-mm` overrides it for a NEW cache directory -- the name
# is derived from the parameters, so a 160 mm cache can never be written into,
# or resumed against, the verified 130 mm one.
CROP_MM = 130.0
IMG = 336
N_SLICES = 12
# Fraction of the ordered stack to sample across. INHERITED from the public
# baseline. TESTED 2026-08-28 and LOST: the 0.06-0.94 cache (`_b06-94`) scored
# T1 -0.0086 on fold 0, and lost on the lateral labels it targeted. Confound:
# N_SLICES is fixed, so a wider band spreads the SAME 12 slices over 1.9x the
# anatomy and halves density through the joint centre. A fair test needs more
# slices, i.e. a rebuild -- not this flag. (This line used to say "never
# tested" and led to the band being re-recommended as untested, repeatedly.) The 0.935 notebook uses 0.06-0.94 and states why:
# "the outer slices are where the collateral ligaments and the lateral meniscus
# sit, and cutting them was measurably costing accuracy on exactly those
# findings". On a sagittal stack ordered medial->lateral that is exactly right --
# the lateral meniscus is at the lateral extreme, the MCL at the medial one --
# and those are two of our three worst labels. Overridable via --band.
BAND = (0.20, 0.80)
PCT_LO, PCT_HI = 1.0, 99.0
FATSAT_SOURCE = "provided"      # vs "recovered"; see module docstring
MIN_SLICE_FRAC = 0.5            # a slot needs this share of its picks readable

# name, plane, accepted weightings, fat-suppression requirement
SLOTS = [
    ("SAG_FLUID_FS",   "Sagittal", ("PD", "T2"), True),
    ("COR_FLUID_FS",   "Coronal",  ("PD", "T2"), True),
    ("AX_FLUID_FS",    "Axial",    ("PD", "T2"), True),
    ("SAG_FLUID_NOFS", "Sagittal", ("PD", "T2"), False),
    # T1 is an explicit class, never "not fluid-sensitive": the negation also
    # admits GRE and unknown, which is 26.9% of everything otherwise eligible.
    ("COR_T1",         "Coronal",  ("T1",),      False),
    ("SAG_T1",         "Sagittal", ("T1",),      False),
]
N_SLOT = len(SLOTS)

# Half-field-of-view crops centred on ONE compartment instead of the joint
# centre. Used with a SMALLER --crop-mm: the box is shifted by (130 - crop)/2 so
# it still lies inside the verified 130 mm field, and the same 336 px then cover
# half the anatomy -- finer sampling at identical train and inference cost.
COMPARTMENTS = {"medial": "med", "lateral": "lat"}


def cache_dir(crop_mm: float | None = None,
              band: tuple[float, float] | None = None,
              native_nb: bool = False, n_slices: int | None = None,
              compartment: str = "", img_px: int | None = None) -> Path:
    """Where a cache with these parameters lives. Derived, never hand-written."""
    n = f"cache_slots_{int(img_px or IMG)}_{int(crop_mm or CROP_MM)}mm_{n_slices or N_SLICES}sl"
    b = tuple(band or BAND)
    if b != BAND:                      # default band keeps the existing name,
        n += f"_b{int(b[0]*100):02d}-{int(b[1]*100):02d}"   # so 10ddbb187a stands
    if native_nb:                      # native-neighbour triplets: (6,12,3,H,W)
        n += "_nb"
    if compartment:                    # half-FOV crop on one compartment
        if compartment not in COMPARTMENTS:
            raise ValueError(f"compartment must be one of {sorted(COMPARTMENTS)}")
        n += f"_{COMPARTMENTS[compartment]}"
    return ROOT / n


# Kept so existing callers and the shipped notebook keep resolving `OUT`.
OUT = cache_dir()


def fingerprint(crop_mm: float | None = None,
                band: tuple[float, float] | None = None,
                native_nb: bool = False, n_slices: int | None = None,
                compartment: str = "", img_px: int | None = None) -> dict:
    """Every choice that decides the pixels, plus a digest of them.

    The directory name records resolution, crop and slice count. It does NOT
    record the band, the percentile window, the fat-suppression policy or the
    version of the sequence parser -- and none of those changes a single shape.
    A cache built under one reading and resumed under another therefore loads
    cleanly, trains, and differs only in a number no output of the run reveals.
    So the manifest carries all of them and resume refuses on any difference.
    """
    spec = {
        "crop_mm": float(crop_mm or CROP_MM), "img": int(img_px or IMG), "n_slices": n_slices or N_SLICES, "band": list(band or BAND),
        "pct": [PCT_LO, PCT_HI], "fatsat_source": FATSAT_SOURCE,
        "min_slice_frac": MIN_SLICE_FRAC,
        "sequence_version": sequence.VERSION,
        "slots": [[n, p, list(w), f] for n, p, w, f in SLOTS],
        # Ordering, laterality and the mirror come from geometry.py; bump this
        # by hand if its rules change, since a code edit there moves pixels too.
        "geometry_version": 1,
    }
    # Added ONLY when on, so the default spec -- and its digest 10ddbb187a --
    # is byte-for-byte what it was before this option existed.
    if native_nb:
        spec["native_nb"] = True
    if compartment:
        spec["compartment"] = compartment
        # Record the shift itself, not just its name: the name alone would not
        # pin the pixels if the default crop ever moved.
        spec["compartment_offset_mm"] = (CROP_MM - float(crop_mm or CROP_MM)) / 2.0
        # Sagittal slots are exempt and keep the full box; without this the digest
        # would claim every slot is a half-FOV crop.
        spec["compartment_sagittal_crop_mm"] = CROP_MM
    spec["digest"] = hashlib.md5(
        json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]
    return spec


def check_manifest(out_dir: Path, force: bool, crop_mm: float | None = None,
                   band: tuple[float, float] | None = None,
                   native_nb: bool = False, n_slices: int | None = None,
                   compartment: str = "", img_px: int | None = None):
    """Write the manifest, or refuse to resume against a different one."""
    fp = fingerprint(crop_mm, band, native_nb, n_slices, compartment, img_px)
    path = out_dir / "manifest.json"
    if path.is_file():
        old = json.loads(path.read_text())
        if old.get("digest") != fp["digest"] and not force:
            diff = {k: (old.get(k), fp[k]) for k in fp
                    if k != "digest" and old.get(k) != fp[k]}
            raise SystemExit(
                f"{path} was built under a different preprocessing spec "
                f"({old.get('digest')} vs {fp['digest']}).\nDiffering: {diff}\n"
                f"Resuming would mix two readings of the same corpus in one "
                f"directory. Use a new directory, or --force to overwrite.")
    path.write_text(json.dumps(fp, indent=2))
    return fp


def band_indices(n: int, band: tuple[float, float] | None = None,
                 n_slices: int | None = None) -> np.ndarray:
    """N_SLICES positions across the central band of an n-slice stack.

    Repeats rather than fails when the band holds fewer than N_SLICES: a thin
    stack is a real acquisition, and refusing it would empty a slot the presence
    mask would then report as absent -- losing a series we can in fact read.
    """
    b = band or BAND
    lo, hi = int(n * b[0]), max(int(n * b[1]), int(n * b[0]) + 1)
    return np.clip(np.round(np.linspace(lo, hi - 1, n_slices or N_SLICES)), 0, n - 1).astype(int)


def crop_square(img: np.ndarray, row_sp: float, col_sp: float,
                crop_mm: float | None = None, offset_mm: float = 0.0):
    """Centre a CROP_MM square on the image centre. Returns (crop, padded).

    Centre rather than foreground centroid: a constant extent with a moving
    centre trades one inconsistency for another, and a knee field of view is
    prescribed around the joint already.

    The crop must be allowed to exceed the image, which it does for 0.54% of
    series. Edge-padding is explicit and flagged; silently returning a smaller
    box is the failure this cache exists to remove -- it would leave those
    series unnormalised with nothing anywhere reporting it.
    """
    crop_mm = float(crop_mm or CROP_MM)
    h, w = img.shape
    nr, nc = int(round(crop_mm / row_sp)), int(round(crop_mm / col_sp))
    r0, c0 = (h - nr) // 2, (w - nc) // 2
    c0 += int(round(offset_mm / col_sp))    # 0 unless this is a compartment crop
    pr0, pc0 = max(0, -r0), max(0, -c0)
    pr1, pc1 = max(0, r0 + nr - h), max(0, c0 + nc - w)
    sub = img[max(r0, 0):min(r0 + nr, h), max(c0, 0):min(c0 + nc, w)]
    if pr0 or pc0 or pr1 or pc1:
        return np.pad(sub, ((pr0, pr1), (pc0, pc1)), mode="edge"), True
    return sub, False


def _interp(h: int, w: int, img_px: int | None = None):
    """INTER_AREA is correct downsampling and wrong upsampling.

    Both axes are tested, not just the width: PixelSpacing is isotropic on
    almost every series here but is not guaranteed to be, and a mixed resize
    (one axis shrinking, one growing) is served by neither pure kernel.
    """
    tgt = int(img_px or IMG)
    if h >= tgt and w >= tgt:
        return cv2.INTER_AREA
    if h <= tgt and w <= tgt:
        return cv2.INTER_CUBIC
    return cv2.INTER_LINEAR


def read_slot(series_dir: Path, laterality: str,
              crop_mm: float | None = None,
              band: tuple[float, float] | None = None,
              native_nb: bool = False, n_slices: int | None = None,
              compartment: str = "", img_px: int | None = None):
    """One slot image: (N_SLICES, IMG, IMG) uint8, canonically oriented.

    Only the sampled slices are decoded -- a 28-slice series costs 12 reads, not
    28 -- and the percentile window is taken over those 12, which is what the
    model sees and so the right stack to normalise against.

    A corrupt slice costs its own position, not the series: the sampler walks
    outward to the nearest readable index instead. `preprocess.build_series_
    volume` already tolerates bad slices this way and the slot path must not
    silently regress on it.
    """
    heads, paths, tag = read_series_headers(series_dir)
    if len(heads) < 2:
        raise ValueError(f"{series_dir.name}: {len(heads)} readable headers")
    g = series_geometry(heads, laterality_tag=laterality)
    order = list(g.order)
    row_sp, col_sp = heads[0].pixel_spacing

    # Compartment crop: shift the box onto one compartment. Canonical orientation
    # puts MEDIAL on the LOW columns (geometry.needs_lr_flip), but the mirror is
    # applied AFTER the resize at the end of this function, so the shift here lives
    # in the PRE-flip native frame -- mirroring reverses the column axis, hence the
    # flip_lr negation. Get that sign wrong and the cache trains on the OPPOSITE
    # compartment while looking perfectly normal, so it is asserted in the builder's
    # verification pass rather than trusted.
    # Sagittal is never mirrored and carries medial->lateral in its SLICE ORDER,
    # not in-plane: cropping it sideways would cut anterior/posterior instead, so
    # sagittal slots stay centred and this stays a single variable.
    off, slot_crop = 0.0, crop_mm
    if compartment and g.plane == "Sagittal":
        # Sagittal carries medial->lateral in its SLICE ORDER, so an in-plane crop
        # here would cut ANTERIOR/POSTERIOR instead -- the patellar tendon and the
        # posterior capsule, i.e. where Baker's and the cruciates live. Keep the
        # verified full-width box so these slots stay BIT-IDENTICAL to the default
        # cache and the compartment change is one variable, not two.
        slot_crop = CROP_MM
    elif compartment:
        off = (CROP_MM - float(crop_mm or CROP_MM)) / 2.0
        off *= 1.0 if compartment == "lateral" else -1.0
        if g.flip_lr:
            off = -off

    cache, dead = {}, set()

    def decode(k):
        """Pixels at ordered position k, or None if that slice will not read."""
        if k in cache:
            return cache[k]
        if k in dead:
            return None
        try:
            a, p = crop_square(_read_pixels(paths[order[k]]), row_sp, col_sp,
                               slot_crop, offset_mm=off)
            cache[k] = (a, p)
            return cache[k]
        except Exception:
            dead.add(k)
            return None

    NS = n_slices or N_SLICES
    out, pos, padded, substituted = [], [], False, 0
    for k in band_indices(len(order), band, NS):
        got, at = decode(int(k)), int(k)
        if got is None:
            # Walk outward for the nearest readable neighbour. Anatomy changes
            # slowly along the stack, so a neighbour is a far better stand-in
            # than a zero frame -- which would read as a black knee.
            for d in range(1, len(order)):
                for cand in (int(k) - d, int(k) + d):
                    if 0 <= cand < len(order) and (got := decode(cand)) is not None:
                        at = cand
                        break
                if got is not None:
                    substituted += 1
                    break
        if got is None:
            continue
        a, p = got
        out.append(a)
        pos.append(at)          # where this centre was ACTUALLY read (after any walk)
        padded |= p

    if len(out) < max(2, MIN_SLICE_FRAC * NS):
        raise ValueError(f"{series_dir.name}: only {len(out)}/{NS} slices readable")
    while len(out) < NS:                        # pad the tail by repetition
        out.append(out[-1])
        pos.append(pos[-1])
    if len({a.shape for a in out}) > 1:
        raise ValueError(f"{series_dir.name}: inconsistent slice shapes")

    vol = np.stack(out).astype(np.float32)
    lo, hi = np.percentile(vol, [PCT_LO, PCT_HI])
    if native_nb:
        # Team-Oxygen-style 2.5D: each centre's TRUE through-plane neighbours
        # (source positions c-1, c+1) instead of the neighbouring CACHED planes,
        # which sit a median 2 source slices (~6.5 mm) away. The window (lo, hi)
        # above is still the one over the 12 CENTRE planes, so the centre channel
        # is bit-identical to the default cache and ONLY the side channels
        # change. A neighbour that is out of range, unreadable or a different
        # shape falls back to the centre -- the same clamp _stack_25d applies at
        # the ends of the stack.
        tri = []
        for a, c in zip(out, pos):
            side = []
            for q in (c - 1, c + 1):
                got = decode(q) if 0 <= q < len(order) else None
                side.append(got[0] if got is not None and got[0].shape == a.shape else a)
            tri.append(np.stack([side[0], a, side[1]]))
        vol = np.stack(tri).astype(np.float32)                 # (12, 3, h, w)
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    tgt = int(img_px or IMG)
    interp = _interp(vol.shape[-2], vol.shape[-1], tgt)
    lead = vol.shape[:-2]
    flat = vol.reshape(-1, *vol.shape[-2:])
    vol = np.stack([cv2.resize(s, (tgt, tgt), interpolation=interp) for s in flat])
    vol = vol.reshape(*lead, tgt, tgt)
    if g.flip_lr:                    # coronal/axial only; see needs_lr_flip
        vol = vol[..., ::-1]         # the W axis, for either layout
    return (np.ascontiguousarray((vol * 255).astype(np.uint8)), g, padded,
            substituted, len(dead))


def slot_candidates(series: pd.DataFrame):
    """Every eligible series per slot, thickest stack first.

    A list rather than a single pick so a corrupt or unreadable best candidate
    falls through to the next instead of emptying the slot.

    No substitution ACROSS predicates. Filling an empty T1 slot from the
    non-fat-sat pool would put one series in two slots and make the presence
    mask assert a sequence that was never acquired -- and the per-diagnosis
    attention would then split across two identical slots, double-weighting one
    acquisition. An empty slot is what an absent slot is.
    """
    out = {}
    for name, plane, wts, fs in SLOTS:
        cand = series[(series.plane == plane) & series.weighting.isin(wts)
                      & (series.fatsat == fs)]
        if len(cand):
            out[name] = cand.sort_values("n_files", ascending=False)
    return out


def study_volume(base: Path, rows: pd.DataFrame, study: str = "",
                 digest: str = "", crop_mm: float | None = None,
                 band: tuple[float, float] | None = None,
                 native_nb: bool = False, n_slices: int | None = None,
                 compartment: str = "", img_px: int | None = None):
    """(vol (6,12,336,336) uint8, mask (6,), meta) for one study directory.

    THE inference entry point. The Kaggle notebook calls this rather than
    reimplementing the pipeline, because every decision here -- slot predicate,
    candidate order, slice band, 130 mm crop, percentile window, resize kernel,
    canonical mirror -- changes the pixels without changing a single shape. A
    reimplementation that drifts on any one of them loads the weights cleanly,
    runs, and returns a submission computed from the wrong image.

    `base` is the directory holding this study's series subdirectories, so the
    same function serves data/train_series/<uid> locally and the competition
    mount at inference.
    """
    # Study-level laterality, from every series, before any slot is chosen:
    # per-series resolution would mirror coronal and axial in opposite
    # directions on the 1.2% of studies whose series disagree.
    per_series = []
    for r in rows.itertuples(index=False):
        heads, _, tag = read_series_headers(base / r.SeriesInstanceUID)
        if heads:
            g = series_geometry(heads, laterality_tag=tag)
            per_series.append((g.center_x, tag))
    if not per_series:
        raise ValueError("no readable series")
    lat, lat_src = resolve_study_laterality(per_series)

    NS = n_slices or N_SLICES
    tgt = int(img_px or IMG)
    vol = np.zeros((N_SLOT, NS, 3, tgt, tgt) if native_nb
                   else (N_SLOT, NS, tgt, tgt), np.uint8)
    mask = np.zeros(N_SLOT, np.float32)
    meta = {"StudyInstanceUID": study, "digest": digest, "laterality": lat,
            "lat_source": lat_src, "resumed": False, "error": None,
            "n_substituted": 0}

    cands = slot_candidates(rows)
    for k, (name, *_) in enumerate(SLOTS):
        for r in cands.get(name, pd.DataFrame()).itertuples(index=False):
            try:
                img, g, padded, subs, ndead = read_slot(
                    base / r.SeriesInstanceUID, lat, crop_mm, band, native_nb,
                    n_slices, compartment, img_px)
            except Exception:
                continue                     # try the next candidate
            vol[k], mask[k] = img, 1.0
            meta[f"{name}__uid"] = r.SeriesInstanceUID
            meta[f"{name}__padded"] = padded
            meta[f"{name}__irregular"] = bool(g.irregular)
            meta[f"{name}__dead_slices"] = ndead
            meta["n_substituted"] += subs
            break
    if mask.sum() == 0:
        raise ValueError("no slot filled")
    meta["n_slots"] = int(mask.sum())
    return vol, mask, meta


def build_study(args):
    study, rows, split, digest, crop_mm, out_root, band, native_nb, n_slices, compartment, img_px = args
    dst = Path(out_root) / split / f"{study}.pt"
    if dst.exists():
        # Metadata is stored INSIDE the .pt, so a resumed study returns the same
        # row it returned when it was built. Returning a bare stub instead would
        # silently drop n_slots, the chosen UIDs and the padding flags for every
        # study carried over from an earlier batch -- which the documented
        # `--limit 40` pilot then full build triggers by construction.
        try:
            meta = torch.load(dst, map_location="cpu", mmap=True,
                              weights_only=False)["meta"]
            meta["resumed"] = True
            return meta
        except Exception:
            dst.unlink(missing_ok=True)         # unreadable: rebuild it

    try:
        # One definition, shared with the Kaggle notebook -- see study_volume.
        vol, mask, meta = study_volume(
            ROOT / "data" / f"{split}_series" / study, rows, study, digest,
            crop_mm, band, native_nb, n_slices, compartment, img_px)
        meta["split"] = split

        tmp = dst.with_suffix(".tmp")
        torch.save({"vol": torch.from_numpy(vol),
                    "mask": torch.from_numpy(mask), "meta": meta}, tmp)
        tmp.replace(dst)
        return meta
    except Exception:
        return {"StudyInstanceUID": study, "split": split, "resumed": False,
                "n_slots": 0, "error": traceback.format_exc(limit=1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=0,
                    help="build only the first N studies, for visual inspection")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--force", action="store_true",
                    help="overwrite a manifest built under a different spec")
    ap.add_argument("--band", type=float, nargs=2, default=list(BAND),
                    metavar=("LO", "HI"),
                    help="fraction of the ordered stack to sample across; "
                         "writes to its OWN directory when not the default")
    ap.add_argument("--crop-mm", type=float, default=CROP_MM,
                    help="physical crop extent; writes to its OWN directory, so "
                         "an existing cache at another crop is never touched")
    ap.add_argument("--native-nb", action="store_true",
                    help="store each centre's true source neighbours (6,12,3,H,W); "
                         "writes to its OWN directory (suffix _nb)")
    ap.add_argument("--n-slices", type=int, default=N_SLICES,
                    help="centre planes per slot; writes to its OWN directory when not 12")
    ap.add_argument("--img", type=int, default=IMG,
                    help="output pixels per side; writes to its OWN directory when not 336. "
                         "The cache is built from the ORIGINAL DICOMs at this size -- training a\n"
                         "bigger model against a smaller cache would silently measure upsampling.")
    ap.add_argument("--compartment", default="", choices=["", *COMPARTMENTS],
                    help="crop a half field of view onto one compartment (coronal/axial "
                         "only; sagittal stays centred). Requires --crop-mm below 130; "
                         "writes to its OWN directory (suffix _med / _lat)")
    a = ap.parse_args()

    band = tuple(a.band)
    if a.compartment and a.crop_mm >= CROP_MM:
        raise SystemExit(f"--compartment needs --crop-mm below {CROP_MM:.0f}; a full-width "
                         f"box cannot be shifted onto one compartment and stay inside the FOV")
    out_root = cache_dir(a.crop_mm, band, a.native_nb, a.n_slices, a.compartment, a.img)
    out_dir = out_root / a.split  # split-specific: no UID collisions, and the
    out_dir.mkdir(parents=True, exist_ok=True)   # size report means something
    fp = check_manifest(out_root, a.force, a.crop_mm, band, a.native_nb, a.n_slices,
                        a.compartment, a.img)
    print(f"spec {fp['digest']}: {a.img}px / {a.crop_mm:.0f}mm / {a.n_slices}sl "
          f"/ band {band} / fatsat={FATSAT_SOURCE}  ->  {out_root.name}")
    print(f"  pixel pitch {a.crop_mm/a.img:.3f} mm/px"
          f"{'  *** ABOVE the 0.5 mm Nyquist bound ***' if a.crop_mm/a.img > 0.5 else ''}")

    hdr_path = ROOT / f"artifacts/slot_headers_{a.split}.parquet"
    if not hdr_path.is_file():
        raise SystemExit(f"{hdr_path} missing -- run first:\n"
                         f"  python -m rsna.data.audit_slots --split {a.split} --dump")
    prov = pd.read_csv(ROOT / f"data/{a.split}_series.csv")
    hdr = pd.read_parquet(hdr_path)
    df = prov.merge(hdr[["SeriesInstanceUID", "plane", "weighting", "n_files"]],
                    on="SeriesInstanceUID", how="inner")
    df["fatsat"] = df.Fat_Suppression.astype(bool)

    studies = sorted(df.StudyInstanceUID.unique())
    if a.limit:
        studies = studies[:a.limit]
    keep = set(studies)                          # built once, not per group
    jobs = [(s, g, a.split, fp["digest"], a.crop_mm, str(out_root), band,
             a.native_nb, a.n_slices, a.compartment, a.img)
            for s, g in df.groupby("StudyInstanceUID") if s in keep]
    print(f"building {len(jobs)} studies -> {out_dir}")

    out = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(build_study, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            out.append(f.result())
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    m = pd.DataFrame(out)
    bad = m[m.error.notna()]
    print(f"\nbuilt {len(m) - len(bad)} | resumed {int(m.resumed.sum())} "
          f"| errors {len(bad)}")
    if len(bad):
        print(bad[["StudyInstanceUID", "error"]].head(10).to_string())
    ok = m[m.error.isna()]
    print("slots per study:", ok.n_slots.value_counts().sort_index().to_dict())
    print(f"slices substituted for corrupt neighbours: {int(ok.n_substituted.sum())}")
    for name, *_ in SLOTS:
        col = f"{name}__padded"
        if col in ok:
            print(f"  {name:15s} filled {int(ok[col].notna().sum()):5d} "
                  f"| edge-padded {int(ok[col].fillna(False).sum())}")
    # Per-cache meta: this line used to write slot_meta_{split}.parquet for EVERY
    # cache variant, so any non-default build overwrote the default's record.
    sfx = ("" if out_root == cache_dir() else "_nb" if out_root == cache_dir(native_nb=True)
           else f"__{out_root.name}")          # existing names kept; every other variant its own
    m.to_parquet(ROOT / f"artifacts/slot_meta_{a.split}{sfx}.parquet", index=False)
    gb = sum(p.stat().st_size for p in out_dir.glob("*.pt")) / 1024 ** 3
    print(f"cache size ({a.split}): {gb:.1f} GiB  in {out_root}")


if __name__ == "__main__":
    main()
