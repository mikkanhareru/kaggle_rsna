#!/usr/bin/env python
"""Build the 64-slice 'widedense' training corpus for the Dread 0.924 reconstruction (R).
Registered: competition_details/preregistered/2026-09-22_dread_reconstruction.md

The preprocessing is NOT re-typed. It is extracted with `ast` from the pulled reference inference
script (competition_details/reference/dread_0924_inference.py, sha256 in SHA256SUMS) and exec'd as
that exact source text. That script labels the block "verbatim from kprep2/dino_preprocess.py,
retargeted to TEST", i.e. the author's own claim is that it IS the training preprocessing -- so a
corpus built by running it over TRAIN studies matches what inference computes, by construction.

Output (the local single-file layout the training script's _open_corpus() looks for first):
    dread/train/all_vols.npy   (N, 64, 336, 336) uint8
    dread/train/all_masks.npy  (N, 64) uint8
    dread/train/all_ids.npy    (N,) str
    python -m rsna.data.build_dread_corpus [--limit N] [--workers 32]
"""
from __future__ import annotations
import argparse, ast, hashlib, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "competition_details/reference/dread_0924_inference.py"
REF_SHA = "859aae12ecea593b45ab0a5f6463fa573cd3083f340c161e920f10fbba29e81b"
CONSTS = ("IMG", "CROP_MM", "SLOTS", "MAXS")
FUNCS = ("_make_reader", "_pick_series_for_slot", "build_study")
OUT = ROOT / "dread/train"
DATA = ROOT / "data"


def extract_source() -> str:
    """The reference's own constants and preprocessing functions, as their exact source text."""
    raw = REF.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != REF_SHA:
        raise SystemExit(f"reference script changed: sha256 {got} != pinned {REF_SHA}")
    text = raw.decode()
    tree = ast.parse(text)
    seg = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in CONSTS:
            seg[node.targets[0].id] = ast.get_source_segment(text, node)
        elif isinstance(node, ast.FunctionDef) and node.name in FUNCS:
            seg[node.name] = ast.get_source_segment(text, node)
    missing = [k for k in CONSTS + FUNCS if k not in seg]
    if missing:
        raise SystemExit(f"could not extract {missing} from the reference")
    return "\n\n".join(seg[k] for k in CONSTS + FUNCS)


_NS = None
def _worker_init(src):
    global _NS
    import glob, os as _os
    _NS = {"np": np, "glob": glob, "os": _os}
    exec(compile(src, str(REF), "exec"), _NS)
    _NS["_reader"] = _NS["_make_reader"]()


def _build_one(args):
    row, sid, rows = args
    try:
        vol, mask = _NS["build_study"](sid, {sid: rows}, str(DATA / "train_series"), _NS["_reader"])
        V = np.load(OUT / "all_vols.npy", mmap_mode="r+"); V[row] = vol; V.flush(); del V
        return row, int(mask.sum()), None
    except Exception as e:                       # the reference zero-fills on failure; record it
        return row, 0, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    src = extract_source()
    ns = {}; _worker_init(src); ns = _NS
    MAXS, IMG = ns["MAXS"], ns["IMG"]
    print(f"reference sha256 OK | extracted {len(CONSTS)} constants + {len(FUNCS)} functions "
          f"({len(src.splitlines())} lines) | MAXS {MAXS} IMG {IMG} CROP_MM {ns['CROP_MM']}")
    print(f"SLOTS {ns['SLOTS']}")

    ids = pd.read_csv(DATA / "train.csv")["StudyInstanceUID"].astype(str).tolist()
    ser = pd.read_csv(DATA / "train_series.csv")
    ser["StudyInstanceUID"] = ser["StudyInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID")}  # as the reference
    if a.limit: ids = ids[:a.limit]
    N = len(ids)
    OUT.mkdir(parents=True, exist_ok=True)
    V = np.lib.format.open_memmap(OUT / "all_vols.npy", mode="w+", dtype=np.uint8,
                                  shape=(N, MAXS, IMG, IMG)); del V
    print(f"{N} studies -> {OUT}  ({N*MAXS*IMG*IMG/1024**3:.1f} GiB)")

    masks = np.zeros((N, MAXS), np.uint8); errs = []; t0 = time.time()
    jobs = [(i, s, SER.get(s, [])) for i, s in enumerate(ids)]
    with ProcessPoolExecutor(a.workers, initializer=_worker_init, initargs=(src,)) as ex:
        futs = [ex.submit(_build_one, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            row, nv, err = f.result()
            if err: errs.append((ids[row], err))
            if n % 250 == 0 or n == N:
                el = time.time() - t0
                print(f"  {n}/{N}  {el:.0f}s  eta {el/n*(N-n)/60:.1f} min  errors {len(errs)}", flush=True)
    V = np.load(OUT / "all_vols.npy", mmap_mode="r")          # masks from the WRITTEN pixels,
    for i in range(N):                                        # exactly as build_study defines them
        masks[i] = (V[i].reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    np.save(OUT / "all_masks.npy", masks)
    np.save(OUT / "all_ids.npy", np.array(ids, dtype=object), allow_pickle=True)
    nv = masks.sum(1)
    print(f"\nDONE {time.time()-t0:.0f}s | errors {len(errs)} | filled slices/study "
          f"mean {nv.mean():.1f} min {nv.min()} max {nv.max()} | empty studies {(nv==0).sum()}")
    for s, e in errs[:10]: print(f"  ERR {s}: {e}")


if __name__ == "__main__":
    main()
