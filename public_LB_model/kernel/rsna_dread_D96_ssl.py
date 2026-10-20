#!/usr/bin/env python3
"""RGR (2 seeds) + the old pipeline's MAE/SSL member -- the registered shipping blend.

Registered: competition_details/preregistered/2026-09-29_SSL_member.md
    final = 0.4*rank(DENSE96 s42) + 0.4*rank(DENSE96 s43) + 0.2*rank(SSL s4mae)
    The LB-0.935 kernel with the two CoAtNet checkpoints swapped for their DENSE96
    counterparts and the corpus constants raised to 96 slices / native 384 / 86 windows.
    The SSL member and the 0.4/0.4/0.2 weights are UNCHANGED and NOT re-tuned.
    A single DENSE96 seed already read LB 0.935 (vs 0.929 for the 64-slice single seed).

Two independent pipelines run one after the other, each lifted from a notebook that has already scored on the
leaderboard, with the edits listed in the registration and nothing else:
  PHASE 1  the s4mae member, from kaggle_submission_H5.ipynb (LB 0.910). Four edits: s4mae becomes the reference
           bundle, per-column ownership is emptied, the five-member composition is removed, and the main loop
           keeps only this member's mean-of-fold-ranks. Its slot/native-neighbour decode, bundle discovery,
           cache-spec assertions and per-checkpoint fingerprint checks are unchanged.
  PHASE 2  the two RGR arms, from rsna_dread_Rgr_blend.py (LB 0.934 as a 2-seed blend). Two edits: the ARMS list
           keeps seeds 42 and 43, and the output stage blends in the SSL arm at the registered weights.
Phase 1 finishes and frees its models before phase 2 starts; the only names the two share are imports plus the
constants CROP_MM and SLOTS, which phase 2 redefines for itself after phase 1 is done with them.
Offline evidence for the SSL weight: T1 +0.00566 over the RGR control on five folds, CI [+0.00483, +0.00651],
5/5 folds positive; gold58 (host labels) +0.0031. The weight is not re-tuned here.
"""

# ============================================================================================================
# PHASE 1 -- SSL arm (s4mae)
# ============================================================================================================
import sys, os, gc, time, json
from pathlib import Path

# ============================ PRE-REGISTERED =============================
# Stage 1: which model owns which OUTPUT column. men4-all-four is the 0.896
# configuration. Submissions A (drop Lateral Meniscus) and B (+ rest8 on
# Synovitis/Fracture) both REFINED this and both LOST on the LB -- 0.895 and
# 0.893 -- so we are back to the version that actually scored.
# The generalist owns every column no specialist claims. Named so that swapping it
# is ONE auditable line rather than a scatter of string literals -- submission D used
# "fine" (25 epochs), E uses "fine50" (50 epochs, same recipe otherwise).
GENERALIST = "s4mae"   # PATCHED: s4mae is the only bundle mounted
# The specialist is named too, for the same reason as GENERALIST: swapping it is one
# auditable line. E used "meniscus" (25 epochs), F uses "men50" (50 epochs).
SPECIALIST = "s4mae"   # PATCHED: unused, ownership is empty
OUTPUT_OWNERSHIP = {}   # PATCHED: no per-column ownership here; s4mae is one arm of an outer blend
# Stage 2: EQUAL-WEIGHT rank average over FIVE members, ALL twelve columns.
# No sweep, no per-label ownership -- post-hoc column selection is exactly what
# cost us submissions A and B. Equal fifths is the frozen offline function:
#     (own + colrank(cnat) + colrank(ep50nb) + colrank(cnxd3lo) + colrank(s4mae)) / 5
# which is exactly (4*H4 + colrank(s4mae))/5 -- the weight s5_gate.py froze. A post-hoc
# sweep put the optimum at 1/4 and worth +0.00004 more; it is NOT taken.
# A 0.5*ep50 + 0.5*severity variant was specified and scored offline BEFORE this
# notebook and came out WORSE than ep50 alone (-0.000618). It is not here, and
# searching further weights on the same OOF is the -0.25 transfer lane.
MEMBERS = []   # PATCHED: the H5 five-member composition is NOT used; see the outer blend at the end
# H4: H + cnxd3lo (ConvNeXt-Tiny, DINOv3 pretraining, default cache) as a FOURTH family.
# Registered in competition_details/preregistered/2026-09-16_fourth_member.md, gate add4_gate.py:
# H4 - H +0.00105 [+0.00025, +0.00180], 5/5 folds positive; it also beats the duplicate-member
# control (+0.00149), so the gain is cnxd3lo's, not the 1/3 -> 1/4 re-weighting (-0.00044).
# cnxd3lo carries DINOv3 weights: its bundle MUST ship third_party_licenses/DINOv3_LICENSE.md.
# H5: H4 + s4mae as a FIFTH family. s4mae is the ep50nb recipe with the encoder
# initialised from OUR OWN in-domain MAE pretraining (mae_ssl1_s150000.pt, sha 353becb8)
# at encoder_lr_mult 0.081 -- not from stock DINOv2.
# Registered in competition_details/preregistered/2026-09-21_ssl_fivefold_H5.md, gate s5_gate.py:
# H5 - H4 +0.00152 [+0.00092, +0.00210], 5/5 folds positive, ALL TWELVE per-label deltas positive.
# It also beats the DUPLICATE-SEED control on the clean holdout: on fold 4 -- the only fold the SSL
# encoder never saw -- adding s4mae gave +0.00133 while adding a second stock SEED of a member H4
# already contains gave +0.00006. Equal standalone strength (0.9033 vs 0.9027) but lower correlation
# with ep50nb (0.913 vs 0.948), so the gain is DECORRELATION, not a stronger model.
# Unlabelled-exposure check: mean(folds 0-3) - fold 4 = +0.00024, i.e. pretraining on a fold's
# unlabelled images did NOT materially inflate that fold's score.
# s4mae shares ep50nb's native_nb input contract (cache digest 92441bac40), so it adds ZERO decode:
# +18% forward only, ~5.5-5.8 h projected against the 9 h limit.
# Its encoder derives from DINOv2 (Apache-2.0) further pretrained on COMPETITION data. The resulting
# weights are our own work; no competition data is redistributed.
# =========================================================================

import numpy as np, pandas as pd, torch
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import rankdata

def find_bundles(root="/kaggle/input"):
    """Every mounted directory holding a manifest.json, keyed by its run_tag.

    os.walk with followlinks, NOT Path.rglob: rglob does not traverse symlinks,
    and Kaggle exposes a dataset at both /kaggle/input/<slug> and
    /kaggle/input/datasets/<user>/<slug>.
    """
    found = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        if "manifest.json" in filenames:
            found.append(Path(dirpath)); dirnames[:] = []
    out = {}
    for d in sorted(found):
        if not (d / "fold0.pt").is_file():
            continue
        try:
            m = json.loads((d / "manifest.json").read_text())
        except (OSError, ValueError):
            continue
        if "model_fingerprints" in m:
            out.setdefault(m["run_tag"], (d, m))
    return out

BUNDLES = find_bundles()
# Source comes from whichever bundle ships it -- NOT a hardcoded slug. Pinning one
# meant dropping that dataset from the notebook silently broke the import.
#
# BUT: each bundle ships the model.py CURRENT WHEN IT WAS BUILT, and one of them
# defines the model class for ALL of them. The old `next(...)` took whichever
# sorted first, so which version won was decided by a filename ordering. That is
# how `n_regions` -- present only in the ep50 bundle -- produced a TypeError while
# submission C survived only because `cnat` happens to sort first AND happens to
# carry `head_type`. Choose by CAPABILITY instead, and verify it.
NEEDED_KW = ("fine_attn", "head_type", "n_regions")
# H needs the CACHE source too: only builders from the native-neighbour era can decode the _nb
# layout. Scoring model.py alone would tie every recent bundle and let a filename ordering
# pick an old build_slot_cache.py that cannot build what ep50nb was trained on.
NEEDED_CACHE_KW = ("native_nb",)
def _src_score(d):
    try:
        t = (d / "rsna/train/model.py").read_text()
        c = (d / "rsna/data/build_slot_cache.py").read_text()
    except OSError:
        return -1
    return sum(k in t for k in NEEDED_KW) + sum(k in c for k in NEEDED_CACHE_KW)
_cands = [d for d, _ in BUNDLES.values()
          if (d / "rsna/data/build_slot_cache.py").is_file()]
assert _cands, f"no bundle ships the source; mounted: {sorted(BUNDLES)}"
CODE_BUNDLE = max(_cands, key=_src_score)
print(f"source bundle: {CODE_BUNDLE.name} "
      f"(supports {_src_score(CODE_BUNDLE)}/{len(NEEDED_KW) + len(NEEDED_CACHE_KW)} required kwargs)")
sys.path.insert(0, str(CODE_BUNDLE))
# Purge any ALREADY-IMPORTED rsna modules before importing below. sys.path only
# affects modules not yet in sys.modules, so on a re-run of this cell in a kernel
# that already imported `rsna.train.model` from a DIFFERENT bundle, the insert
# above is a no-op and the stale module silently wins. That is exactly how a
# capability score of 3/3 (read from the FILE) can disagree with a signature
# check (read from the CACHED MODULE) -- and the disagreement is the symptom, not
# a contradiction. Cheap, idempotent, and makes this cell safe to re-run.
for _m in [k for k in sys.modules if k == "rsna" or k.startswith("rsna.")]:
    del sys.modules[_m]

from rsna.data.audit_slots import probe
from rsna.data.build_slot_cache import (SLOTS, N_SLOT, study_volume,
                                        fingerprint as cache_fingerprint)
from rsna.train.dataset import (IMAGENET_MEAN, IMAGENET_STD,
                                _sample_indices, _stack_25d)
from rsna.train.fingerprint import check as check_fingerprint
from rsna.train.model import load_for_inference
# Fail HERE, naming the bundle and the missing keyword, rather than as a TypeError
# inside the per-fold load loop where the cause is not visible.
import inspect as _inspect
import rsna.train.model as _model_mod
_missing = [k for k in NEEDED_KW
            if k not in _inspect.signature(load_for_inference).parameters]
assert not _missing, (
    f"load_for_inference is missing {_missing}.\n"
    f"  source bundle chosen : {CODE_BUNDLE.name} (file scores "
    f"{ {d.name: _src_score(d) for d in _cands} })\n"
    f"  module actually loaded from: {getattr(_model_mod, '__file__', '?')}\n"
    f"  If the chosen bundle's FILE has the keyword but the loaded MODULE does "
    f"not, a stale import won -- restart the kernel. Otherwise rebuild that "
    f"bundle from current source.")
assert "native_nb" in _inspect.signature(study_volume).parameters, (
    f"study_volume from {CODE_BUNDLE.name} cannot build the native-neighbour cache -- "
    f"mount a bundle built from native-neighbour-era source (e.g. ep50nb)")
from rsna.labels.schema import COLUMN_MAP
LABELS = list(COLUMN_MAP.values())

assert GENERALIST in BUNDLES, (
    f"generalist bundle {GENERALIST!r} missing; found {sorted(BUNDLES)}")
for _m in MEMBERS:
    assert _m in BUNDLES, f"member {_m!r} missing; mounted: {sorted(BUNDLES)}"
for k in OUTPUT_OWNERSHIP:
    assert k in BUNDLES, f"OUTPUT_OWNERSHIP names {k!r}, mounted: {sorted(BUNDLES)}"

# A SPECIALIST emits twelve logits but only its trained labels got gradient. Its
# manifest records which -- assert the ownership map is a subset, so a bundle swap that
# silently changes the trained label set cannot route noise into an output column.
# PATCHED (edit P1-specialist-check): this block validates a SPECIALIST's per-column ownership. There is no
# specialist here -- OUTPUT_OWNERSHIP is empty and s4mae is a generalist trained on all twelve labels -- so the
# check has nothing to validate and its premise ("the manifest records a trained label subset") does not hold.
# It is skipped explicitly rather than satisfied with a fake specialist. The assertion it protects (ownership
# never routes a column to a model that never trained on it) is vacuously true when ownership is empty, and the
# loop above already asserts every name in OUTPUT_OWNERSHIP is mounted.
assert not OUTPUT_OWNERSHIP, f"a specialist reappeared: {OUTPUT_OWNERSHIP} -- restore the ownership check"
print("no specialist: OUTPUT_OWNERSHIP is empty, every column comes from the generalist (s4mae)")

OWNER = {l: GENERALIST for l in LABELS}
for k, cols in OUTPUT_OWNERSHIP.items():
    for c in cols:
        assert c in LABELS, f"unknown label {c!r}"
        OWNER[c] = k
print("stage 1 -- column ownership:")
for l in LABELS:
    print(f"  {l:18s} <- {OWNER[l]}")
print(f"stage 2 -- equal fifths: owner + {' + '.join(MEMBERS)}, all 12 columns")

def find_comp():
    for c in [Path("/kaggle/input/rsna-knee-abnormality-detection"),
              Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")]:
        if (c / "test.csv").is_file():
            return c
    for d in sorted(p for p in Path("/kaggle/input").iterdir() if p.is_dir()):
        if (d / "test.csv").is_file() and (d / "test_series").is_dir():
            return d
    raise FileNotFoundError("competition mount not found")

COMP = find_comp()
NEED = list(dict.fromkeys([GENERALIST, *OUTPUT_OWNERSHIP.keys(), *MEMBERS]))

# All bundles share ONE cache spec, so each study is decoded once. They do NOT
# share img_size: the DINOv2 bundles are 336 and CoAtNet is 384. The decode is
# identical either way -- only the final interpolation differs -- so we group the
# bundles by their input size and resize once per distinct size, not per model.
# reference spec comes from the GENERALIST, which is always in NEED. A bare
# "fine" here would KeyError when the old bundle is not mounted -- or worse,
# silently adopt ITS spec while the loop below validates a different set.
M0 = BUNDLES[GENERALIST][1]
N_GROUPS = M0["n_groups"]
CROP_MM = float(M0["cache_spec"]["crop_mm"])
# TWO cache layouts, ONE decode. The native-neighbour (_nb) cache stores each of the SAME 12
# centre planes with its true source neighbours, and its centre channel is BIT-IDENTICAL to the
# default cache (verified against both stored caches). So each study is decoded once with
# native_nb=True: default-cache members read vol_nb[:, :, 1], _nb members read the triplets.
NB = {k: bool(BUNDLES[k][1].get("native_nb")) for k in NEED}
ANY_NB = any(NB.values())
SPEC = {False: cache_fingerprint(CROP_MM), True: cache_fingerprint(CROP_MM, native_nb=True)}
assert float(SPEC[False]["crop_mm"]) == CROP_MM, SPEC[False]
spec = SPEC[False]["digest"]
SIZES = {}                     # (img_size, native_nb) -> bundle tags
for k in NEED:
    m = BUNDLES[k][1]
    want = SPEC[NB[k]]["digest"]
    assert m["cache_spec"]["digest"] == want, (
        f"{k}: bundle cache spec {m['cache_spec']['digest']} != runtime {want} "
        f"(native_nb={NB[k]}) -- the bundle was fitted on a different reading")
    assert int(m.get("cache_slices") or 12) == 12, f"{k}: only 12-plane caches are wired here"
    assert tuple(m.get("band") or (0.20, 0.80)) == (0.20, 0.80), f"{k}: only the default band is wired"
    assert m["n_groups"] == N_GROUPS, f"{k} n_groups {m['n_groups']} != {N_GROUPS}"
    SIZES.setdefault((int(m["img_size"]), NB[k]), []).append(k)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = (torch.bfloat16 if DEV == "cuda" and torch.cuda.is_bf16_supported()
       else torch.float16)
# torch.cuda.is_bf16_supported() counts EMULATED bf16, so it is True on Kaggle pre-Ampere
# GPUs (T4 sm75, P100 sm60). Emulated bf16 is fine for matmul -- H scored 0.906 that way --
# but cuDNN ships NO bf16 depthwise-conv engine below sm80, and ConvNeXt-Tiny's 7x7
# depthwise conv then dies with 'GET was unable to find an engine to execute this
# computation'. That killed H4 v1 on Kaggle while all 25 fingerprints passed (those run in
# true fp32, autocast off). Fix: keep bf16 for every member whose kernels exist (the three
# ViT-S/14 DINOv2 members, s4mae included, and the CoAtNet), and drop only the ConvNeXt to
# fp16 where bf16 depthwise convs do not exist. Measured on our own
# fold-0 cnxd3lo weights: fp16 is CLOSER to fp32 than bf16 (max|d| 7.1e-3 vs 5.6e-2, rank
# corr 1.000000 vs 0.993), so this is a fidelity gain, not a compromise.
CAP = torch.cuda.get_device_capability() if DEV == "cuda" else (0, 0)
BF16_CONV_OK = CAP[0] >= 8
AMP_BY = {k: (torch.float16
              if (AMP is torch.bfloat16 and not BF16_CONV_OK
                  and "convnext" in BUNDLES[k][1]["backbone"].lower())
              else AMP) for k in NEED}
FP32_FALLBACK = set()   # members whose kernel is missing entirely; filled loudly below
test = pd.read_csv(COMP / "test.csv")
IS_COMMIT = not bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
pd.DataFrame({"StudyInstanceUID": test.StudyInstanceUID,
              **{l: 0.5 for l in LABELS}}).to_csv("submission.csv", index=False)
print(f"spec default {SPEC[False]['digest']} | _nb {SPEC[True]['digest']} | crop {CROP_MM:.0f}mm "
      f"| groups {N_GROUPS} | one _nb decode per study: {ANY_NB}")
for (s, nbf), ks in sorted(SIZES.items()):
    print(f"  img {s} {'_nb triplets' if nbf else 'default (centre channel)'}: {ks}")
print(f"{len(test)} studies | {'COMMIT' if IS_COMMIT else 'SCORING'} | {DEV} "
      f"sm{CAP[0]}{CAP[1]} | bf16 convs {BF16_CONV_OK} | default {AMP}")
for _k, _v in AMP_BY.items():
    print(f"  {_k}: {_v}" + ("   <-- downcast: no bf16 conv kernel here" if _v is not AMP else ""))

MODELS, MODEL_IMG = {}, {}
for k in NEED:
    d, m = BUNDLES[k]
    img = int(m["img_size"])
    grp = []
    for f in range(m["folds"]):
        ck = d / f"fold{f}.pt"
        mdl = load_for_inference(m["backbone"], img, ck, DEV,
                                 fine_attn=m.get("fine_attn", False),
                                 head_type=m.get("head_type", "hier"),
                                 n_regions=m.get("n_regions") or 0)
        check_fingerprint(mdl, img, m["model_fingerprints"][ck.name], DEV,
                          tag=f"{k} fold{f}: ")
        grp.append(mdl)
    MODELS[k], MODEL_IMG[k] = grp, img
    print(f"  {k:10s} {len(grp)} folds @ {img}px, {m['backbone']}, fingerprints OK")
print(f"total checkpoints: {sum(len(v) for v in MODELS.values())}")

t_hdr = time.time()
ts = pd.read_csv(COMP / "test_series.csv")
jobs = [(r.StudyInstanceUID, r.SeriesInstanceUID,
         str(COMP / "test_series" / r.StudyInstanceUID / r.SeriesInstanceUID))
        for r in ts.itertuples(index=False)]
with ThreadPoolExecutor(16) as ex:
    hdr = pd.DataFrame(list(ex.map(probe, jobs)))
n_bad = int(hdr.error.notna().sum())
hdr = hdr[hdr.error.isna()].drop(columns="error")
df = ts.merge(hdr[["SeriesInstanceUID", "plane", "weighting", "n_files"]],
              on="SeriesInstanceUID", how="inner")
df["fatsat"] = df.Fat_Suppression.astype(bool)
by_study = {s: d for s, d in df.groupby("StudyInstanceUID")}
print(f"header pass: {len(df)}/{len(ts)} series, {n_bad} unreadable, "
      f"{len(by_study)}/{len(test)} studies, {time.time()-t_hdr:.0f}s")

@torch.no_grad()
def predict_study(uid):
    """{bundle: (n_folds, 12) raw logits}. Zeros on failure -- LOGIT space, where
    0.0 is sigmoid 0.5. Writing 0.5 would bias unreadable studies to 0.62."""
    blank = {k: np.zeros((len(v), len(LABELS)), np.float32) for k, v in MODELS.items()}
    rows = by_study.get(uid)
    if rows is None or not len(rows):
        return blank
    vol_nb = None
    try:
        if ANY_NB:
            vol_nb, mask, _ = study_volume(COMP / "test_series" / uid, rows, uid, native_nb=True)
            vol = np.ascontiguousarray(vol_nb[:, :, 1])   # == the default cache, bit-identical
        else:
            vol, mask, _ = study_volume(COMP / "test_series" / uid, rows, uid)
    except Exception:
        # The _nb decode reads more slices. If it fails, keep the default-cache members
        # alive with a plain decode; only the _nb members fall back to blank.
        try:
            vol, mask, _ = study_volume(COMP / "test_series" / uid, rows, uid)
            vol_nb = None
        except Exception:
            return blank
    n_slice = vol.shape[1]
    base = {False: [], True: []}
    slots, pos = [], []
    for k in range(N_SLOT):
        if mask[k] < 0.5:
            continue
        # Deterministic bin centres, NOT the training-time stratified draw. SAME centres
        # for both layouts -- only channels 0 and 2 differ.
        idx = _sample_indices(n_slice, N_GROUPS, False, None)
        base[False].append(_stack_25d(torch.from_numpy(vol[k]), idx).float().div_(255.0))
        if vol_nb is not None:
            base[True].append(torch.from_numpy(vol_nb[k][idx]).float().div_(255.0))
        slots.append(k); pos.append(idx / max(n_slice - 1, 1))
    if not slots:
        return blank
    st = torch.tensor(slots, dtype=torch.long).unsqueeze(0).to(DEV)
    sp = torch.tensor(np.stack(pos), dtype=torch.float32).unsqueeze(0).to(DEV)
    sm = torch.ones(1, len(slots), dtype=torch.bool, device=DEV)

    out = {}
    for (img_size, nbf), tags in SIZES.items():
        if nbf and vol_nb is None:
            continue                  # _nb decode failed: these members stay blank below
        # Resize ONCE per distinct input size and share it across every bundle
        # that wants that size. Normalisation happens after the resize, matching
        # SlotDataset exactly -- doing it before would scale interpolated values.
        xs = []
        for x in base[nbf]:
            y = x
            if y.shape[-1] != img_size:
                y = torch.nn.functional.interpolate(
                    y, size=(img_size, img_size), mode="bilinear", align_corners=False)
            xs.append(y.sub(IMAGENET_MEAN).div(IMAGENET_STD))
        img = torch.stack(xs).unsqueeze(0).to(DEV)
        for k in tags:
            o = []
            for mdl in MODELS[k]:
                try:
                    with torch.autocast(DEV, dtype=AMP_BY[k], enabled=DEV == "cuda"):
                        o.append(mdl(img, st, sp, sm).float().cpu().numpy()[0])
                except RuntimeError as e:
                    # A kernel this GPU lacks. fp32 always exists. Loud, once per member --
                    # a silent dtype change is how a submission computes something else.
                    if k not in FP32_FALLBACK:
                        FP32_FALLBACK.add(k)
                        print(f"  !! {k}: {type(e).__name__} under {AMP_BY[k]} -> fp32 for the rest")
                    AMP_BY[k] = torch.float32
                    o.append(mdl(img, st, sp, sm).float().cpu().numpy()[0])
            out[k] = np.stack(o)
        del img
    # EVERY bundle, always: a missing key would misalign preds[k] across studies.
    return {k: out.get(k, blank[k]) for k in MODELS}


# ---- PHASE 1 main loop: the s4mae member only (H5's within-bundle rank average) -------------
t_ssl = time.time()
_ssl = []
for i, uid in enumerate(test.StudyInstanceUID):
    _ssl.append(predict_study(uid)["s4mae"])
    if i % 100 == 0:
        el = time.time() - t_ssl
        print(f"  [ssl] {i+1}/{len(test)}  {el:.0f}s  eta {el/(i+1)*(len(test)-i-1)/60:.1f} min", flush=True)
    if i % 20 == 0:
        gc.collect()
_A = np.stack(_ssl)                                    # (N, n_folds, 12) raw logits
_nfb = sum(1 for p in _ssl if np.allclose(p, 0.0))
print(f"[ssl] fallback (unreadable) studies: {_nfb}/{len(_ssl)}", flush=True)
assert _nfb / len(test) < 0.05, (_nfb, len(test))
# rank WITHIN each fold, then average the ranks -- H5's rule, verbatim in effect
SSL_RANK = np.stack([[rankdata(_A[:, m, j]) / len(_A) for j in range(len(LABELS))]
                     for m in range(_A.shape[1])]).mean(axis=0).T          # (N, 12)
SSL_IDS = list(test.StudyInstanceUID)
assert SSL_RANK.shape == (len(test), len(LABELS)) and np.isfinite(SSL_RANK).all()
print(f"[ssl] done: {_A.shape[1]} folds averaged | {(time.time()-t_ssl)/60:.1f} min", flush=True)
for _m in list(MODELS):                                # free the SSL models before phase 2
    del MODELS[_m]
MODELS.clear(); gc.collect()
if str(DEV).startswith("cuda"):
    torch.cuda.empty_cache()


# ============================================================================================================
# PHASE 2 -- RGR arms, then PHASE 3 -- the blend
# ============================================================================================================
# %% [code]
#!/usr/bin/env python3
"""(docstring of rsna_dread_Rgr_blend.py removed; see that file)"""
import os, sys, glob, time, json, gc
from concurrent.futures import ThreadPoolExecutor
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
IMG = 384   # DENSE96 native
CROP_MM = 140.0
# 64 slices per study instead of 44, same proportions. Must match the corpus the weights
# were trained on (knee_corpus_v4.py).
SLOTS = [("Sagittal", 1, 27), ("Sagittal", 0, 21), ("Coronal", 1, 18),
         ("Coronal", 0, 12), ("Axial", -1, 18)]   # DENSE96
MAXS = sum(s[2] for s in SLOTS)                     # 64
K_EVAL = 86   # DENSE96: every window position a 96-slot volume holds
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
    # PHASE 2: RGR seeds 42 and 43 only. Seeds 44/45 are dropped: the 4-seed blend scored 0.934 on the
    # public leaderboard, exactly the 2-seed blend's score, so they bought nothing and cost ~1.5 h of the
    # 9 h limit. That budget goes to the SSL arm instead (2026-09-29_SSL_member.md).
    {"file": "raptor_ft_rsnaD96F42_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 1.0,
     "sha256": "902ec7638786ab5c394d37ffafb7ee56601aa716a18f7451328ffa33f2a0ad85"},
    {"file": "raptor_ft_rsnaD96F43_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 1.0,
     "sha256": "222bb6308f716bf340b43f8989e83eab8c15246da3d57a3ce271e9eb005180fb"},
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
    arm_probs = [np.full((N, len(LAB)), 0.5, np.float32) for _ in range(A)]

    # SEQUENTIAL ARMS (the OOM fix): only ONE model is resident at a time, so peak system RAM ==
    # one model == the single-arm champion's footprint (which graded fine at 0.879). Holding both
    # arms simultaneously OOM'd system RAM on the full hidden test. Each study is re-preprocessed
    # per arm (build_study is cheap vs inference) and every per-study buffer is freed. Same models,
    # same windowing, same rank-mean blend -> identical 0.8893 result, just serialized.
    for a, arm in enumerate(ARMS):
        wp = find_weight_file(arm["file"])
        if arm.get("sha256"):          # refuse any file that is not the registered checkpoint
            import hashlib
            _h = hashlib.sha256(open(wp, "rb").read()).hexdigest()
            assert _h == arm["sha256"], f"{arm['file']}: sha256 {_h} != registered {arm['sha256']}"
            print(f"[arm {a}] sha256 OK {_h[:12]} | {wp}", flush=True)
        model, res = load_model(wp, arm["arch"], arm["res"], dev)
        print(f"[arm {a}] loaded {arm['file']} | res {res} | {time.time()-t0:.0f}s", flush=True)
        # BOUNDED PREFETCH: one worker builds study i+1 while this thread does windows+GPU for
        # study i. At most 1 running + 1 queued, so pending work is bounded. Windows stay here.
        # Profiled: preparation is 84% of per-study time, so the GPU was idle most of the run.
        # Verified in rsna/train/prefetch_verify.py on 99 stratified studies with cudnn.benchmark
        # and allow_tf32 matching this script: vol/mask/window-centres bitwise identical 99/99,
        # predictions identical to the serial loop (max|d| 0.0, 0 rank changes) against a
        # serial-vs-serial repeatability baseline, injected whole-study failure keeps the 0.5
        # fallback row and the producer survives. Measured 1.31x wall-clock.
        with ThreadPoolExecutor(max_workers=1) as _ex:
            _fut = _ex.submit(build_study, test_ids[0], SER, tsdir, reader)
            for i, sid in enumerate(test_ids):
                _nxt = _ex.submit(build_study, test_ids[i + 1], SER, tsdir, reader) if i + 1 < N else None
                try:
                    vol, mask = _fut.result()
                    xw = eval_windows(vol, mask, k=K_EVAL, res=res, norm=NORM)
                    arm_probs[a][i] = infer_probs(model, xw, dev)
                    del vol, mask, xw
                except Exception as e:
                    print(f"  [arm {a}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)
                _fut = _nxt
                if (i + 1) % 100 == 0 or i + 1 == N:
                    print(f"  [arm {a}] {i+1}/{N} | {time.time()-t0:.0f}s", flush=True)
        del model
        gc.collect()
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[arm {a}] done + freed | {time.time()-t0:.0f}s", flush=True)

    # ---- PHASE 3: the registered shipping blend -------------------------------------------
    # 0.4*rank(RGR s42) + 0.4*rank(RGR s43) + 0.2*rank(SSL), i.e. the ensemble rule's equal weight
    # per recipe (RGR 0.8 over its two seeds) with the B80 weight registered and validated for the
    # SSL member in competition_details/preregistered/2026-09-29_SSL_member.md
    # (T1 +0.00566 five folds, CI [+0.0048,+0.0065]; gold58 +0.0031). Weights are NOT re-tuned here.
    assert len(ARMS) == 2, len(ARMS)
    assert SSL_IDS == test_ids, "the SSL arm scored a different study order"
    assert SSL_RANK.shape == (N, len(LAB))
    W_RGR, W_SSL = 0.4, 0.2
    assert abs(2 * W_RGR + W_SSL - 1.0) < 1e-12
    parts = [W_RGR * rankpct(np.clip(p, 0, 1)) for p in arm_probs] + [W_SSL * rankpct(SSL_RANK)]
    ranks = sum(parts)
    print(f"[blend] {W_RGR} x 2 RGR seeds + {W_SSL} x SSL (s4mae, 5 folds)", flush=True)
    for _i, _p in enumerate(parts):                       # every arm must move the result
        _alt = sum(q for _j, q in enumerate(parts) if _j != _i)
        assert not np.allclose(ranks, _alt / max(1e-9, 1 - (W_SSL if _i == 2 else W_RGR))), \
            f"arm {_i} changed nothing -- not wired up"
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


# PHASE 2+3 entry point (phase 1 has already produced SSL_RANK / SSL_IDS)
main()
