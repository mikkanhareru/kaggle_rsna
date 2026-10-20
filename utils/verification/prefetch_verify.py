#!/usr/bin/env python
"""Bounded prefetch for the R inference loop + its verification harness. Measures and compares only;
it does not modify any submission. Reference loop semantics preserved exactly, including the
fallback: on any per-study exception the pre-initialised 0.5 row STAYS and the loop continues.

  serial      : reference order, build -> windows -> GPU, one study at a time
  prefetch    : ONE CPU worker builds study i+1 while the main thread does windows+GPU for study i
                (at most 1 running + 1 queued -> bounded); windows stay on the main thread.
Run: python -m rsna.train.prefetch_verify [--n 100]
"""
from __future__ import annotations
import argparse, ast, hashlib, os, resource, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "competition_details/reference/dread_0924_inference.py"
REF_SHA = "859aae12ecea593b45ab0a5f6463fa573cd3083f340c161e920f10fbba29e81b"
C = ("IMG", "CROP_MM", "SLOTS", "MAXS", "K_EVAL", "NORM", "LAB", "_MEAN", "_STD")
FN = ("_make_reader", "_pick_series_for_slot", "build_study", "_eval_centers", "eval_windows",
      "build_backbone", "RaptorClassifier", "load_model", "infer_probs")
FAIL_SENTINEL = "__INJECTED_FAILURE__"


def ref_ns():
    raw = REF.read_bytes(); assert hashlib.sha256(raw).hexdigest() == REF_SHA
    text = raw.decode(); lines = text.splitlines(); tree = ast.parse(text); seg = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in C:
            seg[n.targets[0].id] = ast.get_source_segment(text, n)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in FN:
            st = min([n.lineno] + [d.lineno for d in n.decorator_list]) - 1
            seg[n.name] = "\n".join(lines[st:n.end_lineno])
    import torch.nn as nn, torch.nn.functional as F, timm, glob, gc
    ns = {"np": np, "torch": torch, "nn": nn, "F": F, "timm": timm, "glob": glob, "os": os, "gc": gc}
    exec("\n\n".join(seg[k] for k in C + FN), ns); return ns


def run(ns, model, res, sids, SER, tsdir, reader, prefetch, fail_on=None):
    """Returns (probs, capture, wall). Fallback row = 0.5, exactly as the reference."""
    N = len(sids)
    probs = np.full((N, 12), 0.5, np.float32)
    cap = {}
    def build(sid):
        if fail_on is not None and sid == fail_on:
            raise RuntimeError(FAIL_SENTINEL)
        return ns["build_study"](sid, SER, tsdir, reader)
    def consume(i, sid, vol, mask):
        cap[sid] = dict(vol=hashlib.sha256(np.ascontiguousarray(vol)).hexdigest(),
                        mask=hashlib.sha256(np.ascontiguousarray(mask)).hexdigest(),
                        centers=tuple(ns["_eval_centers"](mask, vol.shape[0], ns["K_EVAL"])))
        xw = ns["eval_windows"](vol, mask, k=ns["K_EVAL"], res=res, norm=ns["NORM"])
        probs[i] = ns["infer_probs"](model, xw, "cuda")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    if not prefetch:
        for i, sid in enumerate(sids):
            try:
                vol, mask = build(sid); consume(i, sid, vol, mask); del vol, mask
            except Exception as e:
                cap[sid] = dict(failed=f"{type(e).__name__}")
    else:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(build, sids[0])
            for i, sid in enumerate(sids):
                nxt = ex.submit(build, sids[i + 1]) if i + 1 < N else None   # <=1 running + 1 queued
                try:
                    vol, mask = fut.result(); consume(i, sid, vol, mask); del vol, mask
                except Exception as e:
                    cap[sid] = dict(failed=f"{type(e).__name__}")
                fut = nxt
    torch.cuda.synchronize()
    return probs, cap, time.perf_counter() - t0


def sample(n):
    ts = pd.read_csv(ROOT/"data/train_series.csv"); ts["StudyInstanceUID"] = ts.StudyInstanceUID.astype(str)
    cnt = ts.groupby("StudyInstanceUID").size()
    ids = [str(x) for x in np.load(ROOT/"dread/train/all_ids.npy", allow_pickle=True)]
    fill = np.load(ROOT/"dread/train/all_masks.npy").sum(1)
    order = np.argsort(fill)
    low = [ids[i] for i in order[:20]]                                   # fewest filled slices
    four = [s for s in cnt[cnt == 4].index if s in set(ids)][:10]        # a slot must be empty
    many = [s for s in cnt[cnt >= 9].index if s in set(ids)][:10]        # most series
    rng = np.random.default_rng(0)
    rest = [s for s in ids if s not in set(low) | set(four) | set(many)]
    rnd = list(rng.choice(rest, size=max(0, n - len(low) - len(four) - len(many)), replace=False))
    out = list(dict.fromkeys(low + four + many + rnd))[:n]
    print(f"sample {len(out)}: {len(low)} low-fill, {len(four)} four-series, {len(many)} many-series, rest random")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--ckpt", default="dread/train/raptor_ft_rsnaR_swa.pt"); a = ap.parse_args()
    # The reference sets these at MODULE level (lines 88-89); extracting only constants and
    # functions skipped them, so the first run of this harness tested under benchmark=False --
    # i.e. MORE deterministic than deployment. Match deployment instead.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    ns = ref_ns(); reader = ns["_make_reader"]()
    ts = pd.read_csv(ROOT/"data/train_series.csv"); ts["StudyInstanceUID"] = ts.StudyInstanceUID.astype(str)
    SER = {k: v.to_dict("records") for k, v in ts.groupby("StudyInstanceUID")}
    tsdir = str(ROOT/"data/train_series"); sids = sample(a.n)
    model, res = ns["load_model"](str(ROOT/a.ckpt), "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", 384, "cuda")
    print(f"cudnn.benchmark={torch.backends.cudnn.benchmark} | K_EVAL={ns['K_EVAL']} | res={res}")

    run(ns, model, res, sids[:3], SER, tsdir, reader, False)       # warm caches + cuDNN autotune
    print("\n--- ABBA timing (serial, prefetch, prefetch, serial) ---")
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    pA, cA, tA = run(ns, model, res, sids, SER, tsdir, reader, False)
    pP, cP, tP = run(ns, model, res, sids, SER, tsdir, reader, True)
    rss_p = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    pP2, cP2, tP2 = run(ns, model, res, sids, SER, tsdir, reader, True)
    pB, cB, tB = run(ns, model, res, sids, SER, tsdir, reader, False)
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    ser, pre = (tA + tB) / 2, (tP + tP2) / 2
    print(f"  serial   {tA:6.2f}s , {tB:6.2f}s  -> mean {ser:6.2f}s  ({ser/len(sids)*1000:.0f} ms/study)")
    print(f"  prefetch {tP:6.2f}s , {tP2:6.2f}s  -> mean {pre:6.2f}s  ({pre/len(sids)*1000:.0f} ms/study)")
    print(f"  SPEEDUP {ser/pre:.2f}x  ({(1-pre/ser)*100:.1f}% wall-clock saved)")
    print(f"  peak RSS  before {rss0:.2f} GiB -> after prefetch {rss_p:.2f} -> end {rss1:.2f} GiB")
    print(f"  peak CUDA allocated {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB")

    print("\n--- INPUT PARITY (must be bitwise) ---")
    bad = [s for s in sids if cA[s] != cP[s]]
    print(f"  vol sha / mask sha / window centres identical: {len(sids)-len(bad)}/{len(sids)}")
    for s in bad[:5]: print(f"    MISMATCH {s[-12:]}: serial {cA[s]} vs prefetch {cP[s]}")

    print("\n--- GPU REPEATABILITY vs PREFETCH EFFECT (raw predictions) ---")
    def cmp(x, y, name):
        d = np.abs(x - y); rk = lambda P: np.stack([pd.Series(P[:, j]).rank().values for j in range(12)], 1)
        nrank = int((rk(x) != rk(y)).sum())
        print(f"  {name:28s} max|d| {d.max():.3e}  mean|d| {d.mean():.3e}  rank changes {nrank}/{x.size}")
        return d.max()
    d_rep = cmp(pA, pB, "serial vs serial (baseline)")
    d_pre = cmp(pA, pP, "serial vs prefetch")
    cmp(pP, pP2, "prefetch vs prefetch")

    print("\n--- FAILURE ISOLATION (inject a whole-study failure) ---")
    vic = sids[len(sids)//2]
    fS, kS, _ = run(ns, model, res, sids, SER, tsdir, reader, False, fail_on=vic)
    fP, kP, _ = run(ns, model, res, sids, SER, tsdir, reader, True, fail_on=vic)
    i = sids.index(vic)
    ok_row = np.allclose(fS[i], 0.5) and np.allclose(fP[i], 0.5)
    others = [j for j in range(len(sids)) if j != i]
    ok_align = np.array_equal(fS[others], fP[others])
    ok_rest = np.allclose(fP[others], pP[others])
    print(f"  victim row is the 0.5 fallback in BOTH arms: {ok_row}")
    print(f"  all other rows identical serial-vs-prefetch : {ok_align}")
    print(f"  all other rows unchanged vs the clean run   : {ok_rest}")
    print(f"  producer survived (studies after the failure scored): "
          f"{not np.allclose(fP[i+1:], 0.5) if i+1 < len(sids) else 'n/a'}")

    ok = (not bad) and ok_row and ok_align and ok_rest and d_pre <= max(d_rep, 1e-6) and pre < ser
    print(f"\nVERDICT: {'ADOPT' if ok else 'DO NOT ADOPT'}  "
          f"(inputs bitwise-identical, fallback preserved, prediction diff within GPU repeatability, faster)")


if __name__ == "__main__":
    main()
