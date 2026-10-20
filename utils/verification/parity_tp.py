"""(1) kernel-code parity on gold58 vs the trainer's saved SWA gold predictions (K=24);
(2) H100 forward throughput, CoAtNet-2 vs RN50D, at K in {86,43,24} (ratio only -- Kaggle is a T4)."""
import ast, sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm, gc
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
R = "/import/fox/will/graduation_thesis/RSNA_kaggle/"
KER = R + "kaggle_kernel_dreadRnD96Ssl/rsna_dread_RnD96_ssl.py"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

text = open(KER).read(); lines = text.splitlines(); ns = {"np": np, "torch": torch, "nn": nn, "F": F, "timm": timm, "gc": gc}
want_c = {"NORM", "_MEAN", "_STD", "LAB"}
want_f = {"build_backbone", "RaptorClassifier", "load_model", "_eval_centers", "eval_windows", "infer_probs"}
got = set()
for n in ast.parse(text).body:
    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in want_c:
        exec(ast.get_source_segment(text, n), ns); got.add(n.targets[0].id)
    elif isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in want_f and n.name not in got:
        st = min([n.lineno] + [d.lineno for d in n.decorator_list]) - 1
        exec("\n".join(lines[st:n.end_lineno]), ns); got.add(n.name)
assert got == want_c | want_f, (want_c | want_f) - got

x = torch.randn(4096, 4096, device="cuda"); torch.cuda.synchronize(); assert torch.isfinite(x @ x).all(); del x
print("GPU real alloc+matmul OK", torch.cuda.get_device_name(0))

ids = np.load(R + "dread/train/all_ids_d96.npy", allow_pickle=True).astype(str)
assert (ids == np.load(R + "dread/train/all_ids.npy", allow_pickle=True).astype(str)).all()
id2row = {u: i for i, u in enumerate(ids)}
vols = np.load(R + "dread/train/all_vols_d96.npy", mmap_mode="r"); masks = np.load(R + "dread/train/all_masks_d96.npy")

def macro(P, Y):
    return np.mean([roc_auc_score(Y[:, j], P[:, j]) for j in range(Y.shape[1]) if 0 < Y[:, j].sum() < len(Y)])

print("\n== (1) parity: kernel code on corpus rows, K=24, vs trainer gold npz ==")
for tag in ["rsnaD96F42", "rsnaRnD96F42"]:
    z = np.load(R + f"dread/train/raptor_gold_{tag}_swa.npz", allow_pickle=True)
    model, res = ns["load_model"](R + f"dread/train/raptor_ft_{tag}_swa.pt", None, 384, "cuda")
    P = np.stack([ns["infer_probs"](model, ns["eval_windows"](np.asarray(vols[id2row[u]]), masks[id2row[u]], k=24, res=res), "cuda")
                  for u in z["ids"].astype(str)])
    d = np.abs(P - z["pred"]); rho = min(spearmanr(P[:, j], z["pred"][:, j])[0] for j in range(12))
    print(f"{tag:14s} arch {type(model.backbone).__name__:10s} | max|d| {d.max():.4f} mean|d| {d.mean():.5f} | "
          f"min col spearman {rho:.4f} | gold58 macro kernel {macro(P, z['truth']):.4f} vs trainer {macro(z['pred'], z['truth']):.4f}")
    del model; gc.collect(); torch.cuda.empty_cache()

print("\n== (2) H100 forward, fp16 autocast as infer_probs, ms/study (median of 15) ==")
tp = {}
for arch in ["coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "resnet50d.ra2_in1k"]:
    bb = ns["build_backbone"](arch); m = ns["RaptorClassifier"](bb, F_dim=bb.num_features).eval().cuda()
    for K in [86, 43, 24]:
        xw = torch.randn(K, 3, 384, 384)
        for _ in range(4): ns["infer_probs"](m, xw, "cuda")
        ts = []
        for _ in range(15):
            torch.cuda.synchronize(); t = time.perf_counter(); ns["infer_probs"](m, xw, "cuda"); torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1e3)
        tp[(arch[:8], K)] = float(np.median(ts))
    del m, bb; gc.collect(); torch.cuda.empty_cache()
for K in [86, 43, 24]:
    c, r = tp[("coatnet_", K)], tp[("resnet50", K)]
    print(f"K={K:2d}  CoAtNet-2 {c:7.1f} ms | RN50D {r:6.1f} ms | ratio {c / r:.2f}x")
