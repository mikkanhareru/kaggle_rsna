"""SSL plane cache: EVERY plane of the slot series the classifier already selects.

Corpus definition, fixed before building (plan v2 section 3):
  * series selection is REUSED from artifacts/slot_meta_train.parquet -- the same six slot
    series the supervised pipeline picked. No new series types enter the first experiment,
    so no unvetted eligibility rule is needed.
  * folds 0-3 ONLY. Fold 4 is reserved from every SSL and development stage; the builder
    refuses to write a fold-4 study.

Parity rules, so SSL pixels equal downstream pixels where they overlap:
  * the intensity window is the SAME one read_slot uses -- percentiles over the TWELVE
    banded centre planes -- then applied to every plane of that series. Recomputing it over
    all planes would change the window and break overlap parity.
  * identical 130 mm crop, canonical ordering, deterministic mirror (g.flip_lr), resize
    kernel, normalise->resize->*255->uint8 order.
  * "no random flip augmentation" is NOT a ban on this canonicalisation: they are different
    things and the mirror stays.

Stored ONCE per plane (57.0 GiB for folds 0-3); the loader builds 2.5D triplets, so we never
materialise three channels per centre (~3x the disk).
    python -m rsna.data.build_ssl_cache --limit 40
"""
from __future__ import annotations
import argparse, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2, numpy as np, pandas as pd, torch

from rsna.data.build_slot_cache import (SLOTS, CROP_MM, IMG, PCT_LO, PCT_HI, BAND, N_SLICES,
                                        band_indices, crop_square, _interp)
from rsna.data.geometry import series_geometry
from rsna.data.preprocess import _read_pixels, read_series_headers, resolve_study_laterality

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ssl_cache_336_130mm"
HOLDOUT_FOLD = 4                      # reserved from ALL SSL + development


def build_series(series_dir: Path, laterality: str):
    """(planes uint8 (n,IMG,IMG), meta). Window from the 12 centres, applied to all planes."""
    heads, paths, tag = read_series_headers(series_dir)
    if len(heads) < 2:
        raise ValueError(f"{series_dir.name}: {len(heads)} readable headers")
    g = series_geometry(heads, laterality_tag=laterality)
    order = list(g.order)
    row_sp, col_sp = heads[0].pixel_spacing
    cache, dead = {}, set()

    def dec(k):
        if k in cache: return cache[k]
        if k in dead: return None
        try:
            a, _ = crop_square(_read_pixels(paths[order[k]]), row_sp, col_sp, CROP_MM)
            cache[k] = a; return a
        except Exception:
            dead.add(k); return None

    # 1) the window read_slot would use: the twelve banded centres, with its outward walk
    ref = []
    for k in band_indices(len(order), BAND, N_SLICES):
        got, k = dec(int(k)), int(k)
        if got is None:
            for d in range(1, len(order)):
                for cand in (k - d, k + d):
                    if 0 <= cand < len(order) and (got := dec(cand)) is not None: break
                if got is not None: break
        if got is not None: ref.append(got)
    if len(ref) < 2:
        raise ValueError(f"{series_dir.name}: only {len(ref)} centre planes readable")
    lo, hi = np.percentile(np.stack(ref).astype(np.float32), [PCT_LO, PCT_HI])

    # 2) every plane under THAT window
    planes, kept, interp = [], [], None
    for k in range(len(order)):
        a = dec(k)
        if a is None: continue
        v = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        if interp is None: interp = _interp(v.shape[-2], v.shape[-1])
        v = cv2.resize(v, (IMG, IMG), interpolation=interp)
        if g.flip_lr: v = v[:, ::-1]
        planes.append(np.ascontiguousarray((v * 255).astype(np.uint8))); kept.append(k)
    return np.stack(planes), dict(n_src=len(order), kept=kept, lo=float(lo), hi=float(hi),
                                  flip_lr=bool(g.flip_lr), plane=g.plane, dead=len(dead),
                                  centres=[int(x) for x in band_indices(len(order), BAND, N_SLICES)])


def build_study(args):
    study, uids, fold = args
    if fold == HOLDOUT_FOLD:
        return {"StudyInstanceUID": study, "error": "HOLDOUT fold 4 -- refused"}
    dst = OUT / "train" / f"{study}.pt"
    if dst.exists():
        return {"StudyInstanceUID": study, "resumed": True, "n_planes": 0}
    try:
        base = ROOT / "data" / "train_series" / study
        per_series = []
        for r in (base / u for u in uids):
            heads, _, tag = read_series_headers(r)
            if heads: per_series.append((series_geometry(heads, laterality_tag=tag).center_x, tag))
        lat, _ = resolve_study_laterality(per_series)
        out, n = {}, 0
        for uid in uids:
            try:
                v, m = build_series(base / uid, lat)
            except Exception:
                continue
            out[uid] = {"planes": torch.from_numpy(v), "meta": m}; n += len(v)
        if not out: raise ValueError("no series built")
        tmp = dst.with_suffix(".tmp")
        torch.save({"study": study, "fold": int(fold), "series": out}, tmp); tmp.replace(dst)
        return {"StudyInstanceUID": study, "resumed": False, "n_planes": n, "n_series": len(out)}
    except Exception:
        return {"StudyInstanceUID": study, "error": traceback.format_exc(limit=1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="first N eligible studies (QA subset)")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    folds = pd.read_parquet(ROOT / "artifacts/folds.parquet")
    slot = pd.read_parquet(ROOT / "artifacts/slot_meta_train.parquet").set_index("StudyInstanceUID")
    ucols = [f"{n}__uid" for n, *_ in SLOTS if f"{n}__uid" in slot.columns]
    jobs = []
    for s, fold in folds.fold.items():
        if fold == HOLDOUT_FOLD or s not in slot.index: continue
        uids = [u for u in (slot.at[s, c] for c in ucols) if isinstance(u, str)]
        if uids: jobs.append((s, uids, int(fold)))
    if a.limit: jobs = jobs[:a.limit]
    (OUT / "train").mkdir(parents=True, exist_ok=True)
    print(f"building {len(jobs):,} studies (folds 0-3 only; fold {HOLDOUT_FOLD} RESERVED) -> {OUT}")
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(build_study, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 200 == 0: print(f"  {i}/{len(jobs)}", flush=True)
    m = pd.DataFrame(rows)
    bad = m[m.get("error").notna()] if "error" in m else m.iloc[:0]
    print(f"built {len(m)-len(bad)} | errors {len(bad)} | planes {int(m.get('n_planes', pd.Series([0])).sum()):,}")
    if len(bad): print(bad.head(5).to_string())
    gb = sum(p.stat().st_size for p in (OUT/'train').glob('*.pt')) / 1024**3
    print(f"cache size: {gb:.2f} GiB in {OUT}")
    m.to_parquet(ROOT / "artifacts/ssl_meta_train.parquet", index=False)


if __name__ == "__main__":
    main()
