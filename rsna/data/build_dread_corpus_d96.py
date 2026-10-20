#!/usr/bin/env python
"""DENSE96: the registered corpus builder with TWO counted substitutions on the extracted
reference source -- 96 slices (same slot proportions) at NATIVE 384.
Registered: competition_details/preregistered/2026-10-08_DENSE96_native384.md

    SLOTS  18/14/12/8/12  ->  27/21/18/12/18   (x1.5, MAXS 64 -> 96)
    IMG    336            ->  384              (native; removes the 336->384 upsample)
CROP_MM is UNCHANGED, so the field of view is identical and only sampling density changes.

Writes *_d96 files. The production 64-slice corpus is never touched, and every output path is
guarded -- the base builder has NO overwrite guard and would silently destroy it.
    python -m rsna.data.build_dread_corpus_d96 [--limit N] [--workers 32]
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd

import rsna.data.build_dread_corpus as B          # reuse the registered extraction + worker

SUB = [
 ('SLOTS = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),\n'
  '         ("Coronal", 0, 8), ("Axial", -1, 12)]',
  'SLOTS = [("Sagittal", 1, 27), ("Sagittal", 0, 21), ("Coronal", 1, 18),\n'
  '         ("Coronal", 0, 12), ("Axial", -1, 18)]'),
 ("IMG = 336", "IMG = 384"),
]
OUT = B.OUT
NAMES = ("all_vols_d96.npy", "all_masks_d96.npy", "all_ids_d96.npy")


def d96_source() -> str:
    src = B.extract_source()                      # asserts the reference sha256
    for old, new in SUB:
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"!! expected exactly 1 match of {old[:40]!r}, found {n}")
        src = src.replace(old, new)
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    for n in NAMES:                                # the guard the base builder lacks
        if (OUT / n).exists():
            raise SystemExit(f"!! {OUT/n} exists -- outputs are never overwritten")
    for n in ("all_vols.npy", "all_masks.npy", "all_ids.npy"):
        if not (OUT / n).exists():
            raise SystemExit(f"!! the production corpus {n} is missing -- refusing to run")

    src = d96_source()
    B._worker_init(src); ns = B._NS
    MAXS, IMG = ns["MAXS"], ns["IMG"]
    assert MAXS == 96 and IMG == 384, f"substitution did not take: MAXS {MAXS} IMG {IMG}"
    win = sum(k - 2 for *_, k in ns["SLOTS"])
    print(f"reference sha256 OK | MAXS {MAXS} IMG {IMG} CROP_MM {ns['CROP_MM']} (unchanged)")
    print(f"SLOTS {ns['SLOTS']}")
    print(f"within-slot 3-slice windows: {win}  (the 64-slice corpus yields "
          f"{sum(k-2 for *_, k in [('',1,18),('',0,14),('',1,12),('',0,8),('',-1,12)])})")

    ids = pd.read_csv(B.DATA / "train.csv")["StudyInstanceUID"].astype(str).tolist()
    ser = pd.read_csv(B.DATA / "train_series.csv")
    ser["StudyInstanceUID"] = ser["StudyInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID")}
    if a.limit: ids = ids[:a.limit]
    N = len(ids)
    V = np.lib.format.open_memmap(OUT / NAMES[0], mode="w+", dtype=np.uint8,
                                  shape=(N, MAXS, IMG, IMG)); del V
    print(f"{N} studies -> {OUT/NAMES[0]}  ({N*MAXS*IMG*IMG/1024**3:.1f} GiB)")

    def _build_one(args):                          # same as the base, writing the d96 file
        row, sid, rows = args
        try:
            vol, mask = B._NS["build_study"](sid, {sid: rows}, str(B.DATA / "train_series"),
                                             B._NS["_reader"])
            Vm = np.load(OUT / NAMES[0], mmap_mode="r+"); Vm[row] = vol; Vm.flush(); del Vm
            return row, int(mask.sum()), None
        except Exception as e:
            return row, 0, f"{type(e).__name__}: {e}"

    from concurrent.futures import ProcessPoolExecutor, as_completed
    import rsna.data.build_dread_corpus_d96 as SELF
    masks = np.zeros((N, MAXS), np.uint8); errs = []; t0 = time.time()
    jobs = [(i, s, SER.get(s, [])) for i, s in enumerate(ids)]
    with ProcessPoolExecutor(a.workers, initializer=B._worker_init, initargs=(src,)) as ex:
        futs = [ex.submit(SELF._build_one_mp, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            row, nv, err = f.result()
            if err: errs.append((ids[row], err))
            if n % 250 == 0 or n == N:
                el = time.time() - t0
                print(f"  {n}/{N}  {el:.0f}s  eta {el/n*(N-n)/60:.1f} min  errors {len(errs)}", flush=True)
    V = np.load(OUT / NAMES[0], mmap_mode="r")
    for i in range(N):
        masks[i] = (V[i].reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    np.save(OUT / NAMES[1], masks)
    np.save(OUT / NAMES[2], np.array(ids, dtype=object), allow_pickle=True)
    nv = masks.sum(1)
    print(f"\nDONE {time.time()-t0:.0f}s | errors {len(errs)} | filled slices/study "
          f"mean {nv.mean():.1f} min {nv.min()} max {nv.max()} | empty studies {(nv==0).sum()}")
    for s, e in errs[:10]: print(f"  ERR {s}: {e}")


def _build_one_mp(args):                           # module-level: picklable for ProcessPoolExecutor
    row, sid, rows = args
    try:
        vol, mask = B._NS["build_study"](sid, {sid: rows}, str(B.DATA / "train_series"),
                                         B._NS["_reader"])
        Vm = np.load(OUT / NAMES[0], mmap_mode="r+"); Vm[row] = vol; Vm.flush(); del Vm
        return row, int(mask.sum()), None
    except Exception as e:
        return row, 0, f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    main()
