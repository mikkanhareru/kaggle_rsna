"""Process-pool study preparation for the single-arm DENSE96 kernel: parity + timing + failure isolation.
Measures only; modifies no submission. Functions/constants are exec'd from the scored LB-0.935 kernel.
  thread : the deployed loop -- ONE thread builds study i+1 while the main thread does windows+GPU for i
  pool   : W fork()ed processes build up to 2W studies ahead; windows+GPU stay on the main thread.
           On BrokenProcessPool (a worker died) the remaining studies are built IN-PROCESS, i.e. the
           deployed semantics, so a dead worker can never turn the rest of the test set into 0.5 rows.
Run pinned to 4 CPUs (Kaggle GPU box): taskset -c 0-3 python poolverify.py --n 99"""
import argparse, ast, glob, hashlib, os, time, resource, multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError as FuturesTimeout
from concurrent.futures.process import BrokenProcessPool
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm, gc
ROOT = "/import/fox/will/graduation_thesis/RSNA_kaggle/"
KER = ROOT + "kaggle_kernel_dreadD96/rsna_dread_D96.py"
WANT = {"IMG", "CROP_MM", "SLOTS", "MAXS", "K_EVAL", "NORM", "LAB", "_MEAN", "_STD", "_make_reader", "_pick_series_for_slot",
        "build_study", "_eval_centers", "eval_windows", "build_backbone", "RaptorClassifier", "load_model", "infer_probs"}
OPT = {"PF_THREADS", "_selected_files", "_read_all", "_prefetch"}   # present only in prefetch kernels
MAIN_PID = os.getpid()
_G = {}

def kernel_ns(ker):
    text = open(ker).read(); lines = text.splitlines(); got = set()
    ns = {"np": np, "torch": torch, "nn": nn, "F": F, "timm": timm, "gc": gc, "glob": glob, "os": os,
          "_TPE": ThreadPoolExecutor, "_PG": {}}
    for n in ast.parse(text).body:
        nm = n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else getattr(n, "name", None)
        if nm in (WANT | OPT) and nm not in got:
            st = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])]) - 1
            exec("\n".join(lines[st:n.end_lineno]), ns); got.add(nm)
    assert WANT <= got, WANT - got
    return ns

def _pool_build(sid):
    if _G.get("kill") == sid and os.getpid() != MAIN_PID:
        os._exit(1)                                          # injected HARD worker death (OOM-kill analogue)
    if _G.get("hang") == sid and os.getpid() != MAIN_PID:
        time.sleep(10**6)                                    # injected HANG (deadlocked worker analogue)
    if _G.get("fail") == sid:
        raise RuntimeError("__INJECTED_FAILURE__")
    if _G.get("pf"):
        _G["ns"]["_prefetch"](sid)                           # the kernel's own prefetch, exactly as shipped
    return _G["ns"]["build_study"](sid, _G["SER"], _G["tsdir"], _G["reader"])

def run(model, res, sids, mode, W=3, fail=None, kill=None, hang=None, timeout=300):
    ns = _G["ns"]; _G.update(fail=fail, kill=kill, hang=hang)
    N = len(sids); probs = np.full((N, 12), 0.5, np.float32); cap = {}; note = ""
    def consume(i, sid, vol, mask):
        cap[sid] = dict(vol=hashlib.sha256(np.ascontiguousarray(vol)).hexdigest(),
                        mask=hashlib.sha256(np.ascontiguousarray(mask)).hexdigest(),
                        centers=tuple(ns["_eval_centers"](mask, vol.shape[0], ns["K_EVAL"])))
        probs[i] = ns["infer_probs"](model, ns["eval_windows"](vol, mask, k=ns["K_EVAL"], res=res, norm=ns["NORM"]), "cuda")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    if mode == "thread":
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_pool_build, sids[0])
            for i, sid in enumerate(sids):
                nxt = ex.submit(_pool_build, sids[i + 1]) if i + 1 < N else None
                try:
                    vol, mask = fut.result(); consume(i, sid, vol, mask); del vol, mask
                except Exception as e:
                    cap[sid] = dict(failed=type(e).__name__)
                fut = nxt
    else:
        ex = ProcessPoolExecutor(max_workers=W, mp_context=mp.get_context("fork"))
        depth = 2 * W; futs = {}; broken = False
        for j in range(min(N, depth)):
            futs[j] = ex.submit(_pool_build, sids[j])
        for i, sid in enumerate(sids):
            try:
                # refill FIRST: a study that raises must not skip the submit of study i+depth
                if not broken and i + depth < N:
                    try:
                        futs[i + depth] = ex.submit(_pool_build, sids[i + depth])
                    except BrokenProcessPool:
                        broken = True; note = f"pool broke at submit {i + depth}; rest built in-process"
                if not broken:
                    try:
                        vol, mask = futs.pop(i).result(timeout=timeout)
                    except (BrokenProcessPool, FuturesTimeout) as e:
                        broken = True; note = f"{type(e).__name__} at study {i}; rest built in-process"
                        for p in list((getattr(ex, "_processes", None) or {}).values()):
                            try: p.terminate()
                            except Exception: pass
                        ex.shutdown(wait=False, cancel_futures=True)
                        vol, mask = _pool_build(sid)
                else:
                    vol, mask = _pool_build(sid)
                consume(i, sid, vol, mask); del vol, mask
            except Exception as e:
                cap[sid] = dict(failed=type(e).__name__)
        if not broken: ex.shutdown(wait=True, cancel_futures=True)
    torch.cuda.synchronize()
    return probs, cap, time.perf_counter() - t0, note

def sample(n):
    ts = pd.read_csv(ROOT + "data/train_series.csv"); ts["StudyInstanceUID"] = ts.StudyInstanceUID.astype(str)
    cnt = ts.groupby("StudyInstanceUID").size()
    ids = [str(x) for x in np.load(ROOT + "dread/train/all_ids_d96.npy", allow_pickle=True)]; sid_set = set(ids)
    fill = np.load(ROOT + "dread/train/all_masks_d96.npy").sum(1)
    low = [ids[i] for i in np.argsort(fill)[:20]]
    four = [s for s in cnt[cnt == 4].index if s in sid_set][:10]
    many = [s for s in cnt[cnt >= 9].index if s in sid_set][:10]
    taken = set(low) | set(four) | set(many)
    rnd = list(np.random.default_rng(0).choice([s for s in ids if s not in taken], size=max(0, n - len(taken)), replace=False))
    out = list(dict.fromkeys(low + four + many + rnd))[:n]
    print(f"sample {len(out)}: {len(low)} low-fill, {len(four)} four-series, {len(many)} many-series, rest random | "
          f"cpus visible {len(os.sched_getaffinity(0))}")
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=99); ap.add_argument("--W", type=int, default=3)
    ap.add_argument("--ker", default=KER); ap.add_argument("--pf", action="store_true")
    a = ap.parse_args()
    import cv2; cv2.setNumThreads(1)        # parent, before ANY cv2 use and before fork (the EFF kernel's order)
    torch.backends.cudnn.benchmark = True; torch.backends.cuda.matmul.allow_tf32 = True
    ns = kernel_ns(a.ker)
    assert (not a.pf) or ns.get("PF_THREADS", 0) > 0, "--pf needs a kernel with PF_THREADS > 0"
    ts = pd.read_csv(ROOT + "data/train_series.csv"); ts["StudyInstanceUID"] = ts.StudyInstanceUID.astype(str)
    ts["SeriesInstanceUID"] = ts.SeriesInstanceUID.astype(str)
    _G.update(ns=ns, SER={k: v.to_dict("records") for k, v in ts.groupby("StudyInstanceUID")},
              tsdir=ROOT + "data/train_series", reader=ns["_make_reader"](), pf=a.pf)
    ns["_PG"].update(SER=_G["SER"], tsdir=_G["tsdir"])
    sids = sample(a.n)
    model, res = ns["load_model"](ROOT + "dread/train/raptor_ft_rsnaD96F42_swa.pt", None, 384, "cuda")
    print(f"kernel {a.ker.split('/')[-1]} | K_EVAL={ns['K_EVAL']} IMG={ns['IMG']} MAXS={ns['MAXS']} res={res} W={a.W} "
          f"prefetch={a.pf} (PF_THREADS={ns.get('PF_THREADS', 0)})")
    run(model, res, sids[:4], "thread")                                         # warm caches + cuDNN autotune
    pA, cA, tA, _ = run(model, res, sids, "thread")
    pP, cP, tP, _ = run(model, res, sids, "pool", a.W)
    pP2, cP2, tP2, _ = run(model, res, sids, "pool", a.W)
    pB, cB, tB, _ = run(model, res, sids, "thread")
    ser, poo = (tA + tB) / 2, (tP + tP2) / 2
    print("\n--- ABBA timing (thread, pool, pool, thread), 4 pinned CPUs, warm page cache ---")
    print(f"  thread {tA:6.1f}s, {tB:6.1f}s -> {ser / len(sids) * 1000:.0f} ms/study")
    print(f"  pool   {tP:6.1f}s, {tP2:6.1f}s -> {poo / len(sids) * 1000:.0f} ms/study")
    print(f"  SPEEDUP {ser / poo:.2f}x | peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2:.2f} GiB")
    bad = [s for s in sids if cA[s] != cP[s] or cA[s] != cP2[s]]
    print(f"\n--- INPUT PARITY --- vol/mask sha + window centres identical {len(sids) - len(bad)}/{len(sids)}")
    rk = lambda P: np.stack([pd.Series(P[:, j]).rank().values for j in range(12)], 1)
    def cmp(x, y, name):
        d = np.abs(x - y); print(f"  {name:24s} max|d| {d.max():.3e} rank changes {int((rk(x) != rk(y)).sum())}/{x.size}"); return d.max()
    d_rep = cmp(pA, pB, "thread vs thread (base)"); d_p = cmp(pA, pP, "thread vs pool"); cmp(pP, pP2, "pool vs pool")
    vic = sids[len(sids) // 2]; i = sids.index(vic); oth = [j for j in range(len(sids)) if j != i]
    fP, _, _, _ = run(model, res, sids, "pool", a.W, fail=vic)
    ok_fail = np.allclose(fP[i], 0.5) and np.allclose(fP[oth], pP[oth])
    print(f"\n--- FAILURE 1: exception in a worker --- victim 0.5 and every other row unchanged: {ok_fail}")
    kP, _, _, note = run(model, res, sids, "pool", a.W, kill=vic)
    ok_kill = np.allclose(kP, pP) and bool(note)
    print(f"--- FAILURE 2: worker process DIES --- {note!r} | all rows equal the clean run (victim rebuilt in-process): {ok_kill}")
    t = time.perf_counter(); hP, _, _, hnote = run(model, res, sids, "pool", a.W, hang=vic, timeout=20)
    ok_hang = np.allclose(hP, pP) and "Timeout" in hnote
    print(f"--- FAILURE 3: worker HANGS --- {hnote!r} | all rows equal the clean run: {ok_hang} | {time.perf_counter()-t:.0f}s")
    ok = not bad and d_p <= max(d_rep, 1e-6) and ok_fail and ok_kill and ok_hang and poo < ser
    print(f"\nVERDICT: {'ADOPT' if ok else 'DO NOT ADOPT'}")

if __name__ == "__main__":
    main()
