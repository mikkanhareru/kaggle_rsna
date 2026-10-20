"""Derive the EFF kernel from the scored single-arm d96 kernel (LB 0.935) by asserted substitutions.
usage: build_eff3.py OUT_DIR SLUG CKPT_FILE ARCH SHA256 K_EVAL BUNDLE [W|auto] [PF_THREADS]
Changes: (1) docstring prefix, (2) K_EVAL, (3) ARMS, (4) module-level pool helpers, (5) the per-study loop
becomes the process-pool loop verified in poolverify.py (refill first; BrokenProcessPool -> in-process)."""
import json, os, sys
OUT, SLUG, CK, ARCH, SHA, K, BUNDLE = sys.argv[1:8]; K = int(K)
WARG = sys.argv[8] if len(sys.argv) > 8 else "auto"; PF = int(sys.argv[9]) if len(sys.argv) > 9 else 0
W_EXPR = "max(1, min(4, len(os.sched_getaffinity(0)) - 1))" if WARG == "auto" else str(int(WARG))
R = "/import/fox/will/graduation_thesis/RSNA_kaggle/"
assert not os.path.exists(R + OUT), f"{OUT} exists"
s = open(R + "kaggle_kernel_dreadD96/rsna_dread_D96.py").read()
def sub(old, new):
    global s
    n = s.count(old); assert n == 1, f"{n} hits: {old[:60]!r}"; s = s.replace(old, new)

sub('"""Knee MRI: twelve findings from a single model\n',
    f'"""EFF: the Efficiency-Prize candidate (2026-10-19_EFF_efficiency_track.md).\n'
    f'    {CK} ({ARCH}), K_EVAL {K}, process pool W={WARG}, prefetch threads {PF}, one replica per GPU (v3). Derived from the scored single-arm\n'
    f'    DENSE96 kernel (LB 0.935) by asserted substitutions; the text below describes the parent.\n\n'
    'PARENT: Knee MRI: twelve findings from a single model\n')
sub("K_EVAL = 86   # DENSE96: every window position a 96-slot volume holds (was 42 at 64 slots)\n",
    f"K_EVAL = {K}   # EFF (registered 2026-10-19): was 86, every window position a 96-slot volume holds\n")
sub('    {"file": "raptor_ft_rsnaD96F42_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 1.0},\n',
    f'    {{"file": "{CK}", "arch": "{ARCH}", "res": 384, "w": 1.0,\n     "sha256": "{SHA}"}},\n')
sub("\ndef main():\n", '''
# ---- EFF: process-pool preparation helpers (module level so fork()ed workers can unpickle them) ----
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
from concurrent.futures.process import BrokenProcessPool
POOL_TIMEOUT_S = 300   # watchdog: a healthy pool returns a study in seconds; a hung one falls back to serial
PF_THREADS = 0   # >0: each worker first reads the study's selected series into the page cache (bytes unchanged)
_PG = {}
from concurrent.futures import ThreadPoolExecutor as _TPE
def _selected_files(sid):
    rows = _PG["SER"].get(sid, []); used = set(); files = []
    for plane, fluid, k in SLOTS:                      # the exact series choice build_study will make
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            continue
        used.add(r["SeriesInstanceUID"])
        files += glob.glob(f"{_PG['tsdir']}/{sid}/{r['SeriesInstanceUID']}/*.dcm")
    return files
def _read_all(f):
    try:
        with open(f, "rb") as fh:
            while fh.read(1 << 20):
                pass
    except Exception:
        pass
def _prefetch(sid):
    try:
        with _TPE(PF_THREADS) as tp:
            list(tp.map(_read_all, _selected_files(sid)))
    except Exception:
        pass                                          # best effort: build_study below is unchanged either way
def _pool_build(sid):
    if PF_THREADS > 0:
        _prefetch(sid)
    return build_study(sid, _PG["SER"], _PG["tsdir"], _PG["reader"])


def main():
''')
old_loop_start = "    reader = _make_reader()\n    N = len(test_ids); A = len(ARMS)\n"
old_loop_end = '        print(f"[arm {a}] done + freed | {time.time()-t0:.0f}s", flush=True)\n'
i0, i1 = s.index(old_loop_start), s.index(old_loop_end) + len(old_loop_end)
assert s.count(old_loop_start) == 1 and s.count(old_loop_end) == 1 and i0 < i1
s = s[:i0] + '''    reader = _make_reader()
    N = len(test_ids); A = len(ARMS)
    assert A == 1, "EFF is single-arm: the pool is created once, BEFORE the model touches CUDA (fork safety)"
    arm_probs = [np.full((N, len(LAB)), 0.5, np.float32) for _ in range(A)]

    # EFF PROCESS-POOL PREPARATION. W fork()ed workers build up to 2W studies ahead; windows + GPU stay on this
    # thread. The window is refilled BEFORE the current result is taken, so a study that raises cannot starve a
    # later one. If a worker DIES (BrokenProcessPool) every remaining study is built in-process -- the scored
    # serial semantics -- so a dead worker can never turn the rest of the test set into 0.5 rows. A worker that
    # HANGS trips the POOL_TIMEOUT_S watchdog: the pool is terminated and the rest is built in-process too.
    # cv2 threads are set to 1 HERE, in the parent, before fork and before any cv2 call; workers inherit it.
    # (Calling cv2.setNumThreads inside a forked child after the parent's cv2 pool was live deadlocks -- reproduced.)
    # Verified (scratchpad poolverify.py, 99 stratified studies, 4 pinned CPUs): see the registration.
    import cv2
    cv2.setNumThreads(1)
    _PG.update(SER=SER, tsdir=tsdir, reader=reader)
    W = {W_EXPR}
    ex = ProcessPoolExecutor(max_workers=W, mp_context=mp.get_context("fork"))
    depth = 2 * W; futs = {}; broken = False
    for j in range(min(N, depth)):
        futs[j] = ex.submit(_pool_build, test_ids[j])

    a, arm = 0, ARMS[0]
    wp = find_weight_file(arm["file"])
    import hashlib, queue, threading
    _h = hashlib.sha256(open(wp, "rb").read()).hexdigest()
    assert _h == arm["sha256"], f"{arm['file']}: sha256 {_h} != registered {arm['sha256']}"

    # EFF v3: ONE MODEL REPLICA PER GPU (the Kaggle T4 box has two). Study i -> GPU i mod NG, one thread per GPU.
    # Same checkpoint on every replica, so predictions differ from single-GPU only by fp16 device noise.
    # EFF_FORCE_DEVS / EFF_TEST_FAIL_DEV are LOCAL TEST HOOKS, unset on Kaggle: two replicas on one device, and an
    # injected failure on one replica, so dispatch / retry / retirement can be exercised on a 1-GPU machine.
    if os.environ.get("EFF_FORCE_DEVS"):
        devs = os.environ["EFF_FORCE_DEVS"].split(",")
    elif str(dev).startswith("cuda"):
        devs = [f"cuda:{g}" for g in range(max(1, min(2, torch.cuda.device_count())))]
    else:
        devs = ["cpu"]
    models = []
    for d in devs:
        m_, res = load_model(wp, arm["arch"], arm["res"], d)
        models.append(m_)
    if len(models) > 1:                                  # cross-GPU self-check on a fixed synthetic input
        try:
            xs = torch.randn(K_EVAL, 3, res, res, generator=torch.Generator().manual_seed(0))
            outs = [infer_probs(m_, xs, d) for m_, d in zip(models, devs)]
            dmax = float(max(np.abs(o - outs[0]).max() for o in outs))
            print(f"[gpu] self-check over {devs}: max|d| {dmax:.2e}", flush=True)
            if not dmax <= 1e-2:
                raise RuntimeError(f"replicas disagree by {dmax}")
        except Exception as e:
            print(f"[gpu] self-check FAILED ({type(e).__name__}: {e}); using one GPU", flush=True)
            models, devs = models[:1], devs[:1]
    NG = len(models)
    fail_dev = int(os.environ.get("EFF_TEST_FAIL_DEV", "-1"))
    qs = [queue.Queue() for _ in range(NG)]              # unbounded: a retry can never block a GPU thread
    CAP = 2 * NG + 2                                     # in-flight studies (memory bound) -- held until FINAL
    sem = threading.BoundedSemaphore(CAP)
    alive = [True] * NG; strikes = [0] * NG; lock = threading.Lock()

    def _score(g, i, sid, vol, mask, retried):
        try:
            if g == fail_dev:
                raise RuntimeError("injected device failure (local test hook)")
            xw = eval_windows(vol, mask, k=K_EVAL, res=res, norm=NORM)
            arm_probs[a][i] = infer_probs(models[g], xw, devs[g])
            with lock:
                strikes[g] = 0
            sem.release()
        except Exception as e:
            with lock:
                strikes[g] += 1
                if strikes[g] >= 3 and alive[g] and sum(alive) > 1:
                    alive[g] = False
                    print(f"  [gpu {g}] retired after 3 consecutive failures", flush=True)
                other = [h for h in range(NG) if h != g and alive[h]]
            if not retried and other:
                qs[other[0]].put((i, sid, vol, mask, True))   # retry ONCE on the other GPU; permit still held
            else:
                print(f"  [arm {a}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)
                sem.release()

    def _gpu_worker(g):
        while True:
            item = qs[g].get()
            if item is None:
                break
            _score(g, *item)

    threads = [threading.Thread(target=_gpu_worker, args=(g,), daemon=True) for g in range(NG)]
    for t_ in threads:
        t_.start()
    print(f"[arm {a}] loaded {arm['file']} sha {_h[:12]} | res {res} | K_EVAL {K_EVAL} | pool W={W} | gpus {devs} | {time.time()-t0:.0f}s", flush=True)
    for i, sid in enumerate(test_ids):
        try:
            if not broken and i + depth < N:
                try:
                    futs[i + depth] = ex.submit(_pool_build, test_ids[i + depth])
                except BrokenProcessPool:
                    broken = True
                    print(f"  [pool] broken at submit {i + depth}; building the rest in-process", flush=True)
            if not broken:
                try:
                    vol, mask = futs.pop(i).result(timeout=POOL_TIMEOUT_S)
                except (BrokenProcessPool, FuturesTimeout) as e:
                    broken = True
                    print(f"  [pool] {type(e).__name__} at study {i}; building the rest in-process", flush=True)
                    for p in list((getattr(ex, "_processes", None) or {}).values()):
                        try:
                            p.terminate()
                        except Exception:
                            pass
                    ex.shutdown(wait=False, cancel_futures=True)
                    vol, mask = build_study(sid, SER, tsdir, reader)
            else:
                vol, mask = build_study(sid, SER, tsdir, reader)
            if not sem.acquire(timeout=POOL_TIMEOUT_S):
                print(f"  [gpu] backlog stalled > {POOL_TIMEOUT_S}s at study {i}; dispatching anyway", flush=True)
            with lock:
                live = [h for h in range(NG) if alive[h]] or [0]
            qs[live[i % len(live)]].put((i, sid, vol, mask, False))
            del vol, mask
        except Exception as e:
            print(f"  [arm {a}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)
        if (i + 1) % 100 == 0 or i + 1 == N:
            print(f"  [arm {a}] {i+1}/{N} dispatched | {time.time()-t0:.0f}s", flush=True)
    for _ in range(CAP):                                 # every permit back == every study FINAL (retries included)
        if not sem.acquire(timeout=POOL_TIMEOUT_S):
            print("  [gpu] drain timed out; unfinished studies keep the 0.5 fallback row", flush=True)
            break
    for g in range(NG):
        qs[g].put(None)
    for t_ in threads:
        t_.join(timeout=POOL_TIMEOUT_S)
    if not broken:
        ex.shutdown(wait=True, cancel_futures=True)
    del models
    gc.collect()
    if str(dev).startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"[arm {a}] done + freed | {time.time()-t0:.0f}s", flush=True)
''' + s[i1:]
for _a, _b in [("PF_THREADS = 0   #", f"PF_THREADS = {PF}   #"), ("    W = {W_EXPR}\n", f"    W = {W_EXPR}\n")]:
    assert s.count(_a) == 1, _a; s = s.replace(_a, _b)
os.makedirs(R + OUT)
code = f"rsna_dread_{SLUG.replace('-', '_')}.py"
open(R + OUT + "/" + code, "w").write(s)
m = json.load(open(R + "kaggle_kernel_dreadD96/kernel-metadata.json"))
m.update(id=f"pareru806/rsna-knee-{SLUG}", title=f"RSNA Knee {SLUG.replace('-', ' ')}", code_file=code, dataset_sources=[BUNDLE])
json.dump(m, open(R + OUT + "/kernel-metadata.json", "w"), indent=2)
import ast; ast.parse(s); print("built", OUT, code, "| K_EVAL", K, "|", CK)
