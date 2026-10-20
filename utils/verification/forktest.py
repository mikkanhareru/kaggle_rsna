import sys, multiprocessing as mp, torch
from concurrent.futures import ProcessPoolExecutor
case = sys.argv[1]
def init_torch(): torch.set_num_threads(1)
def init_cv2():
    import cv2; cv2.setNumThreads(1)
def init_none(): pass
def work(x):
    import numpy as np, cv2
    a = np.random.rand(512, 512).astype("float32")
    return float(cv2.resize(a, (384, 384), interpolation=cv2.INTER_AREA).sum())
if "omp" in case:                      # parent uses OpenMP BEFORE fork (what poolverify's warm-up did)
    x = torch.randn(86, 3, 384, 384); (x - 0.5).mul_(2.0).sum()
if "cv2p" in case:                     # parent itself ran cv2 (its parallel_for pool is live)
    import cv2, numpy as np
    for _ in range(20): cv2.resize(np.random.rand(2048, 2048).astype("float32"), (384, 384), interpolation=cv2.INTER_AREA)
if "cv2t" in case:                     # parent ran cv2 inside a worker THREAD, exactly like the prefetch run
    import cv2, numpy as np
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(1) as t:
        list(t.map(lambda _: cv2.resize(np.random.rand(2048, 2048).astype("float32"), (384, 384), interpolation=cv2.INTER_AREA), range(20)))
if "cuda" in case:
    torch.zeros(1, device="cuda")
init = {"torch": init_torch, "cv2": init_cv2, "none": init_none}[case.split("_")[-1]]
with ProcessPoolExecutor(3, mp_context=mp.get_context("fork"), initializer=init) as ex:
    print(case, "OK", len(list(ex.map(work, range(6)))), flush=True)
