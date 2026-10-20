# %% [code]
#!/usr/bin/env python3
"""EFF: the Efficiency-Prize candidate (2026-10-19_EFF_efficiency_track.md).
    raptor_ft_rsnaD96F42_swa.pt (coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k), K_EVAL 43, process pool W=6, prefetch threads 8, one replica per GPU (v3). Derived from the scored single-arm
    DENSE96 kernel (LB 0.935) by asserted substitutions; the text below describes the parent.

PARENT: Knee MRI: twelve findings from a single model

This notebook takes a knee MRI study and scores twelve findings at once: ACL tear, MCL tear,
medial and lateral meniscus tears, osteoarthritis in the medial, lateral and patellofemoral
compartments, joint effusion, synovitis, a Baker's cyst, bone contusion and fracture. It scores
0.924 on the public leaderboard using one model, with no ensembling and no test-time augmentation.

This is the inference half of the work. The model was trained separately and its weights are
attached as a dataset, so this notebook only loads them and predicts:
https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense

Where the training labels came from

Worth saying up front, because it shapes everything else. The competition gives you 4,407 studies
but structured labels for only 58 of them. Every other study arrives with a free-text radiology
report and nothing more, so there is very little to train against out of the box.

The labels behind these weights were made by reading those reports with a language model and
turning each into twelve probabilities rather than twelve yes or no answers. A report that says a
tear is suspected becomes a number near 0.8, not a 1, which is a fairer target than forcing every
hedged sentence into a hard label. That yields 4,349 studies to train on. The 58 studies that came
with real labels were never trained on and are used to check the result honestly; the model reaches
0.9167 macro-AUC on them.

Building a fixed input from studies that are all shaped differently

The hard part of this competition is not the network, it is that no two studies look alike. A
study holds several DICOM series shot in different planes, the number of series varies, and the
number of slices in a series varies more. Anything that expects a fixed-size input has to be given
one.

The approach here is to fill five fixed slots per study, always in the same order, for a stack of
64 images:

  18 slices from a sagittal series, preferring a fluid-sensitive one
  14 slices from a second sagittal series, preferring one that is not fluid-sensitive
  12 slices from a coronal series, preferring a fluid-sensitive one
   8 slices from a second coronal series
  12 slices from an axial series

Preferring a fluid-sensitive series for some slots and not for others is deliberate. Fluid-
sensitive sequences show swelling, effusion and acute injury clearly, while the other sequences
show anatomy and cartilage better, and the twelve findings are split across both. If a study has
no series for a slot, the slot is left as zeros and the model is told to skip it rather than being
fed something misleading.

Within a series, slices are taken evenly across 6 to 94 percent of the stack rather than from the
middle. The outer slices are where the collateral ligaments and the lateral meniscus sit, and
cutting them was measurably costing accuracy on exactly those findings.

Every slice is cropped to a 140 mm box around the centre of the image using the pixel spacing from
the DICOM header, then resized to 336 pixels. Cropping by millimetres rather than by pixel count
matters: it means a knee occupies the same fraction of the frame whether the scan was acquired at
0.3 or 0.5 mm per pixel, so the model is not asked to learn scale differences that carry no medical
information.

How the model reads the stack

Three neighbouring slices are stacked into the three channels of one image. The network then sees
a little of what lies above and below the slice in the middle, which is most of the benefit of a 3D
model at the cost of a 2D one. Each of these three-slice windows is passed through a CoAtNet
backbone at 384 pixels.

The windows are combined with an attention layer that has separate weights for each of the twelve
findings. This is the part that matters most. A cruciate tear may be visible on two sagittal slices
while osteoarthritis is spread across many coronal ones, and a single pooled score forces those two
to share one notion of which slices are important. Giving each finding its own attention weights
lets each one draw on the slices that actually show it.

Running it

Scoring uses 42 windows per study. Inference runs in half precision and automatically retries a
study in full precision if it fails, so no study is ever dropped from the submission. The notebook
needs no internet: the backbone is loaded from the attached weights rather than downloaded.
"""
import os, sys, glob, time, json, gc
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
# T4 (Turing) cuDNN v9 has fp16/fp32 conv engines but NOT bf16 for these shapes
# ("GET was unable to find an engine..."); benchmark lets it pick a valid algo for
# the fixed (1,24,3,res,res) input.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ---- fixed config (must match training exactly) -----------------------------
IMG = 384   # DENSE96: native, no 336->384 upsample
CROP_MM = 140.0
# 64 slices per study instead of 44, same proportions. Must match the corpus the weights
# were trained on (knee_corpus_v4.py).
SLOTS = [("Sagittal", 1, 27), ("Sagittal", 0, 21), ("Coronal", 1, 18),
         ("Coronal", 0, 12), ("Axial", -1, 18)]   # DENSE96: x1.5, MAXS 64 -> 96
MAXS = sum(s[2] for s in SLOTS)                     # 64
K_EVAL = 43   # EFF (registered 2026-10-19): was 86, every window position a 96-slot volume holds
NORM = "imagenet"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Three arms: (weights filename, fallback arch, fallback res). ck carries arch+res too.
# Selected 2026-08-19 by greedy forward selection AND exhaustive subset search over a 7-arm
# panel on the 45-study gold set (phase2/blend_panel.py); both agree on this exact set.
# Singles: coatnet384 0.9025 | swinbase384 0.8825 | effv2l480 0.8716.
# Blend {coatnet+swin+effv2l} = 0.9068 (2-arm {coatnet+swin} = 0.9059, coatnet alone 0.9025).
# Dropped as redundant: cnn336 (0.8833, the former champion), cnbase384 (0.8754),
# cnlarge384 (0.8752), maxvit384 (0.8438).
#
# SINGLE ARM: coatnet_rmlp_2_rw_384 retrained on the EXPANDED 4,349-study corpus.
#
# Why one arm and not the 3-arm blend: on the live leaderboard CoAtNet alone scored 0.914 while
# every blend scored 0.914-0.915, so ensembling is worth ~+0.001 there -- the ~+0.010 it showed
# on the old 45-study gold set was gold-set noise. One arm is also 1/3 the kernel runtime.
#
# Corpus expansion: the corpus previously held 3,200 of the 4,349 labelled studies and only 45
# of the 58 gold studies. Rebuilt to 4,407 studies (+37.8% training data, 58-study gate).
#
# Measured on the 58-study gate (the incumbent re-scored on the SAME gate for a fair compare):
#   incumbent CoAtNet (3,155-study corpus) 0.8923
#   this model       (4,349-study corpus) 0.9054   (+0.0131, better in 92.7% of 2000 bootstraps)
# Biggest gains land on the findings that were capping us: Lateral Meniscus +0.071,
# Fracture +0.057, Lateral OA +0.048, Medial Meniscus +0.035, ACL +0.028.
ARMS = [
    {"file": "raptor_ft_rsnaD96F42_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 1.0,
     "sha256": "902ec7638786ab5c394d37ffafb7ee56601aa716a18f7451328ffa33f2a0ad85"},
]


# ============================================================================
# Model -- verbatim from finetune_raptor.py
# ============================================================================
def build_backbone(arch, pretrained=False):
    # maxvit/maxxvit/coatnet are conv-attention hybrids: NO CLS token, NO interpolatable
    # pos-embed -> avg pool. The "vit" substring in "coatnet"/"maxvit" must NOT route them
    # down the ViT path (mirrors finetune_raptor.py exactly).
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if is_vit:
        kw.update(global_pool="token", dynamic_img_size=True)
    else:
        kw.update(global_pool="avg")
    return timm.create_model(arch, **kw)


class RaptorClassifier(nn.Module):
    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        f = self.backbone(x.flatten(0, 1))
        return f.view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x):
        return self.head(self.encode(x))


def load_model(pt_path, arch_default, res_default, device, ngpu=1):
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", arch_default)
    ck_res = int(ck.get("res", res_default))
    bb = build_backbone(arch, pretrained=False)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device)
    # NOTE: DataParallel removed on purpose. On the full hidden test it drove a system-RAM OOM
    # (per-forward module replication over many studies); a single T4 handles K_EVAL=24 windows
    # fine. Arms are also run SEQUENTIALLY (see main) so peak RAM == one model, not two.
    del ck
    gc.collect()
    return model, ck_res


# ============================================================================
# Eval windowing -- verbatim from finetune_raptor.py StudyWindows (train=False)
# ============================================================================
def _eval_centers(mask, D, k):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]


def eval_windows(vol, mask, k, res, norm=NORM):
    D = vol.shape[0]
    cs = _eval_centers(mask, D, k)
    wins = np.empty((len(cs), 3, res, res), np.float32)
    for j, c in enumerate(cs):
        c = max(1, min(c, D - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(tri)
        if t.shape[-1] != res:
            t = F.interpolate(t[None], size=(res, res), mode="bilinear",
                              align_corners=False)[0]
        wins[j] = t.numpy()
    x = torch.from_numpy(wins)
    if norm == "imagenet":
        x = (x - _MEAN) / _STD
    return x


@torch.no_grad()
def infer_probs(model, xwins, device):
    x = xwins.unsqueeze(0).to(device)
    use_cuda = device != "cpu" and str(device).startswith("cuda")
    if use_cuda:
        # fp16 conv on T4 is fully cuDNN-supported (bf16 is NOT -> "no engine").
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                o = torch.sigmoid(model(x).float())
            return o[0].cpu().numpy()
        except RuntimeError:
            # fp32 always has a Turing conv engine; slower but never drops a study.
            torch.cuda.empty_cache()
            o = torch.sigmoid(model(x).float())
            return o[0].cpu().numpy()
    o = torch.sigmoid(model(x).float())
    return o[0].cpu().numpy()


def rankpct(x):                                   # per-column percentile rank in [0,1]
    order = x.argsort(0).argsort(0).astype(np.float64)
    return order / max(1, (x.shape[0] - 1))


# ============================================================================
# Preprocessing -- verbatim from kprep2/dino_preprocess.py, retargeted to TEST
# ============================================================================
def _make_reader():
    import pydicom, cv2
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    def order_and_meta(sdir):
        fs = glob.glob(sdir + "/*.dcm"); recs = []; ps_list = []
        for f in fs:
            try:
                h = pydicom.dcmread(f, stop_before_pixels=True)
                iop = getattr(h, 'ImageOrientationPatient', None)
                ipp = getattr(h, 'ImagePositionPatient', None)
                if iop is not None and ipp is not None and len(iop) == 6:
                    r = np.array(iop[:3], float); c = np.array(iop[3:], float)
                    n = np.cross(r, c); pos = float(np.dot(np.array(ipp, float), n))
                else:
                    pos = float(getattr(h, 'InstanceNumber', 0) or 0)
                ps = getattr(h, 'PixelSpacing', None); ps = float(ps[0]) if ps is not None else 0.5
                ps_list.append(ps); recs.append((pos, f, ps))
            except Exception:
                recs.append((0.0, f, 0.5))
        recs.sort(key=lambda x: x[0])
        med_ps = float(np.median(ps_list)) if ps_list else 0.5
        return [(f, ps) for _, f, ps in recs], med_ps

    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a

    def mm_crop_resize(a, ps):
        h, w = a.shape; cpx = int(round(CROP_MM / max(ps, 1e-3)))
        cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
        a = a[y0:y0 + cpx, x0:x0 + cpx]
        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)

    return order_and_meta, read_px, mm_crop_resize


def _pick_series_for_slot(rows, plane, fluid, used):
    cands = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get('Fluid_Sensitive', 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_study(sid, ser_records, tsdir, reader):
    order_and_meta, read_px, mm_crop_resize = reader
    rows = ser_records.get(sid, [])
    vol = np.zeros((MAXS, IMG, IMG), np.uint8); idx = 0; used = set()
    for plane, fluid, k in SLOTS:
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k; continue
        used.add(r['SeriesInstanceUID'])
        files, med_ps = order_and_meta(f"{tsdir}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            idx += k; continue
        # wide span: the collateral ligaments and lateral meniscus live in the
        # peripheral slices the old 0.15-0.85 crop threw away. Must match the corpus
        # the weights were trained on (knee_corpus_v2.py, SPAN_LO/SPAN_HI).
        n = len(files); lo, hi = int(n * 0.06), int(n * 0.94) - 1; hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        arrs = []; pss = []
        for p in picks:
            fp, ps = files[min(p, n - 1)]
            try:
                arrs.append(read_px(fp)); pss.append(ps)
            except Exception:
                arrs.append(None); pss.append(med_ps)
        valid = [a for a in arrs if a is not None]
        if valid:
            allpx = np.concatenate([a.ravel() for a in valid])
            loq, hiq = np.percentile(allpx, [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
        for a, ps in zip(arrs, pss):
            if idx >= MAXS: break
            if a is None: idx += 1; continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)
            vol[idx] = (aw * 255).astype(np.uint8); idx += 1
        if idx >= MAXS: break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


# ============================================================================
# Test-root discovery + weights + main
# ============================================================================
def find_test_root():
    cands = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
             "/kaggle/input/rsna-knee-abnormality-detection"]
    for b in cands:
        if os.path.exists(b + "/test.csv"):
            return b
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f and (os.path.isdir(d + "/test_series") or os.path.isdir(d + "/test_images")):
            return d
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f:
            return d
    raise RuntimeError("no test root under /kaggle/input")


def find_weight_file(fname):
    # direct dataset mounts first; NEVER recursive-glob the competitions DICOM tree.
    direct = [f"/kaggle/input/raptor-knee-arms/{fname}",
              f"/kaggle/input/raptor-knee-arms/1/{fname}",
              f"/kaggle/input/raptor-cnn336/{fname}"]
    for p in direct:
        if os.path.exists(p):
            return p
    for d in sorted(glob.glob("/kaggle/input/*/")):
        if "competition" in d.lower():
            continue
        hits = glob.glob(os.path.join(d, "**", fname), recursive=True)
        if hits:
            return hits[0]
    raise RuntimeError(f"{fname} not found under /kaggle/input")


# ---- EFF: process-pool preparation helpers (module level so fork()ed workers can unpickle them) ----
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
from concurrent.futures.process import BrokenProcessPool
POOL_TIMEOUT_S = 300   # watchdog: a healthy pool returns a study in seconds; a hung one falls back to serial
PF_THREADS = 8   # >0: each worker first reads the study's selected series into the page cache (bytes unchanged)
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
    import pandas as pd
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ngpu = torch.cuda.device_count()
    print(f"device {dev} | gpus {ngpu} | torch {torch.__version__}", flush=True)

    ROOT = find_test_root()
    tsdir = ROOT + "/test_series"
    if not os.path.isdir(tsdir):
        tsdir = ROOT + "/test_images"
    print("test root:", ROOT, "| series dir:", tsdir, flush=True)

    test = pd.read_csv(ROOT + "/test.csv"); test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)
    test_ids = test["StudyInstanceUID"].tolist()
    tser = pd.read_csv(ROOT + "/test_series.csv")
    tser["StudyInstanceUID"] = tser["StudyInstanceUID"].astype(str)
    tser["SeriesInstanceUID"] = tser["SeriesInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in tser.groupby("StudyInstanceUID")}
    print(f"test studies {len(test_ids)} | test series {len(tser)}", flush=True)

    sub_cols = ["StudyInstanceUID"] + LAB
    ssub = os.path.join(ROOT, "sample_submission.csv")
    if os.path.exists(ssub):
        sub_cols = list(pd.read_csv(ssub, nrows=1).columns)

    reader = _make_reader()
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
    W = 6
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

    # WEIGHTED rank-mean blend across the test set, per finding (the offline recipe).
    # Weights come from ARMS[*]["w"] and are normalised here, so dropping/adding an arm can
    # never silently change the scale. Falls back to equal weights if none are declared.
    _w = np.array([float(a.get("w", 1.0)) for a in ARMS], dtype=np.float64)
    _w = _w / _w.sum()
    print(f"[blend] weighted rank-mean w={dict(zip([a['file'] for a in ARMS], _w.round(4)))}", flush=True)
    ranks = np.tensordot(_w, np.stack([rankpct(np.clip(p, 0, 1)) for p in arm_probs]),
                         axes=(0, 0))                                          # (N,12) in [0,1]
    if not np.isfinite(ranks).all():
        ranks[~np.isfinite(ranks)] = 0.5

    sub = pd.DataFrame(ranks.astype(np.float32), columns=LAB)
    sub.insert(0, "StudyInstanceUID", test_ids)
    sub = sub[sub_cols]
    assert list(sub.columns) == sub_cols, "column order drift"
    assert sub["StudyInstanceUID"].tolist() == test_ids, "row identity drift"
    assert np.isfinite(sub[LAB].values).all()
    out = "/kaggle/working/submission.csv"
    sub.to_csv(out, index=False)
    print("wrote", out, "|", len(sub), "rows x", len(sub.columns), "cols", flush=True)
    print(sub.head().to_string(index=False), flush=True)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
