"""K_EVAL curve, paired: same checkpoint, same OOF studies, only the number of eval windows changes.
RGR folds 0-4 (CoAtNet-2, res 384) + RN50D folds 0/4 (res from ckpt) on the 64-slice corpus.
Kernel code (eval_windows / infer_probs / load_model) is exec'd from the RND96 kernel source.
Ruler: labels_public_dense binarised at 0.5 on each fold's non-gold OOF studies; gold58 reported separately."""
import ast, json, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, timm, gc
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
R = "/import/fox/will/graduation_thesis/RSNA_kaggle/"
OUT = sys.argv[1]
KS = [62, 42, 24, 12]
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

text = open(R + "kaggle_kernel_dreadRnD96Ssl/rsna_dread_RnD96_ssl.py").read(); lines = text.splitlines()
ns = {"np": np, "torch": torch, "nn": nn, "F": F, "timm": timm, "gc": gc}
want = {"NORM", "_MEAN", "_STD", "LAB", "build_backbone", "RaptorClassifier", "load_model", "_eval_centers", "eval_windows", "infer_probs"}
got = set()
for n in ast.parse(text).body:
    nm = n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else getattr(n, "name", None)
    if nm in want and nm not in got:
        st = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])]) - 1
        exec("\n".join(lines[st:n.end_lineno]), ns); got.add(nm)
assert got == want, want - got
LAB = ns["LAB"]

ids = np.load(R + "dread/train/all_ids.npy", allow_pickle=True).astype(str); id2row = {u: i for i, u in enumerate(ids)}
vols = np.load(R + "dread/train/all_vols.npy", mmap_mode="r"); masks = np.load(R + "dread/train/all_masks.npy")
folds = pd.read_parquet(R + "artifacts/folds.parquet")
lab = pd.read_parquet(R + "dread/labels_public_dense.parquet")
if "StudyInstanceUID" in lab.columns: lab = lab.set_index("StudyInstanceUID")
lab.index = lab.index.astype(str); assert set(LAB) <= set(lab.columns)
gz = np.load(R + "dread/train/raptor_gold_rsnaD96F42_swa.npz", allow_pickle=True)
gold_ids = list(gz["ids"].astype(str)); gold_Y = gz["truth"]
assert set(gold_ids) == set(folds.index[folds.is_gold])

class DS(Dataset):
    def __init__(self, uids, res): self.uids, self.res = uids, res
    def __len__(self): return len(self.uids)
    def __getitem__(self, i):
        r = id2row[self.uids[i]]; v = np.asarray(vols[r]); m = masks[r]
        return {K: ns["eval_windows"](v, m, k=K, res=self.res) for K in KS}

def run(ckpt, uids):
    model, res = ns["load_model"](R + f"dread/train/{ckpt}", None, 384, "cuda")
    dl = DataLoader(DS(uids, res), batch_size=None, num_workers=12, worker_init_fn=lambda _: torch.set_num_threads(1))
    P = {K: np.zeros((len(uids), 12), np.float32) for K in KS}
    for i, w in enumerate(dl):
        for K in KS: P[K][i] = ns["infer_probs"](model, w[K], "cuda")
    del model; gc.collect(); torch.cuda.empty_cache()
    return P, res

def macro(P, Y):
    return float(np.mean([roc_auc_score(Y[:, j], P[:, j]) for j in range(12) if 0 < Y[:, j].sum() < len(Y)]))

x = torch.randn(4096, 4096, device="cuda"); torch.cuda.synchronize(); assert torch.isfinite(x @ x).all(); del x
store, rows = {}, []
plan = [(f"raptor_ft_rsnaRgr{f}_swa.pt", f, "RGR") for f in range(5)] + \
       [("raptor_ft_rsnaRn0hi_swa.pt", 0, "RN50D"), ("raptor_ft_rsnaRn4_swa.pt", 4, "RN50D")]
t0 = time.time()
for ck, f, fam in plan:
    oof = [u for u in folds.index[(folds.fold == f) & ~folds.is_gold] if u in id2row]
    P, res = run(ck, oof + gold_ids)
    Y = (lab.loc[oof, LAB].values >= 0.5).astype(np.float32)
    for K in KS:
        store[f"{fam}_f{f}_K{K}"] = P[K]
        rows.append(dict(fam=fam, fold=f, res=res, K=K, n=len(oof), oof=macro(P[K][:len(oof)], Y),
                         gold=macro(P[K][len(oof):], gold_Y), **{c: roc_auc_score(Y[:, j], P[K][:len(oof), j]) for j, c in enumerate(LAB)}))
    store[f"{fam}_f{f}_ids"] = np.array(oof + gold_ids)
    print(f"[{time.time()-t0:5.0f}s] {fam} fold {f} res {res} n {len(oof)} | " +
          " ".join(f"K{r['K']} {r['oof']:.4f}" for r in rows[-len(KS):]), flush=True)
np.savez(OUT + ".npz", **store)
df = pd.DataFrame(rows); df.to_csv(OUT + ".csv", index=False)
print("\n== paired delta vs K42 (OOF, public ruler), per fold and mean ==")
for fam in ["RGR", "RN50D"]:
    d = df[df.fam == fam].pivot(index="fold", columns="K", values="oof")
    for K in KS:
        dd = d[K] - d[42]; print(f"{fam:5s} K{K:2d}: mean {d[K].mean():.4f} | delta {dd.mean():+.5f}  per-fold " + " ".join(f"{v:+.4f}" for v in dd))
print("\n== per-finding delta K24/K12 vs K42, RGR mean over folds ==")
g = df[df.fam == "RGR"].groupby("K")[LAB].mean()
print((g.loc[[24, 12]] - g.loc[42]).T.round(4).to_string())
print("\n== gold58 (each fold model never saw it; mean of per-fold macro) ==")
print(df.groupby(["fam", "K"]).gold.mean().round(4).to_string())
print(f"DONE {time.time()-t0:.0f}s")
