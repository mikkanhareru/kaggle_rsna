import multiprocessing as mp, cv2, numpy as np, torch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
cv2.setNumThreads(1)                                   # parent, BEFORE any cv2 work and before fork
with ThreadPoolExecutor(1) as t:                       # then the parent uses cv2 in a thread (the poolverify order)
    list(t.map(lambda _: cv2.resize(np.random.rand(2048, 2048).astype("float32"), (384, 384), interpolation=cv2.INTER_AREA), range(20)))
torch.zeros(1, device="cuda"); (torch.randn(86, 3, 384, 384) - 0.5).sum()
def work(x):
    a = np.random.rand(512, 512).astype("float32")
    return cv2.getNumThreads(), float(cv2.resize(a, (384, 384), interpolation=cv2.INTER_AREA).sum()) > 0
with ProcessPoolExecutor(3, mp_context=mp.get_context("fork")) as ex:
    r = list(ex.map(work, range(6)))
print("parent-set, no initializer: OK", "child cv2 threads:", sorted({a for a, _ in r}), flush=True)
