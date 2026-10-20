# Usage:  python rsna/train/loop.py --fold 0 --backbone convnext_tiny.fb_in1k
# Training only. Two-tier evaluation every epoch:
#   Tier 1  OOF vs LLM labels   n~880/fold  SE~0.02  -> SELECTS
#   Tier 2  OOF vs gold labels  n~12/fold   SE~0.06  -> VETOES ONLY
# Promote a change only if it wins Tier 1 and is not worse than noise on Tier 2.

from __future__ import annotations

import argparse, copy, hashlib, json, math, sys, time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rsna.labels.schema import COLUMN_MAP
from rsna.train.augment import (AugConfig, SeriesAugment, assert_no_hflip,
                                config_without)
from rsna.labels.quality import HQ_MIN, high_quality, label_quality
from rsna.train.slot_dataset import SlotDataset
from rsna.train.maxspan_dataset import MaxSpanDataset
from rsna.train.dataset import StudyDataset, collate_studies
from rsna.train.model import KneeModel

LABELS = list(COLUMN_MAP.values())


@dataclass
class Cfg:
    fold: int = 0
    backbone: str = "convnext_tiny.fb_in1k"
    policy: str = "mask"              # baseline; soft_neg is the challenger
    val_policy: str = ""              # "" == cfg.policy. Set it to hold the RULER
                                      # fixed while the TRAINING target changes.
                                      # Required for any policy experiment: the
                                      # sev midpoint reclassifies 1,293 cells
                                      # across evaluate()'s `y > 0.5` line and
                                      # moves the HQ set from 6 labels to 7, so
                                      # --policy alone changes the target AND the
                                      # instrument AND the checkpoint-selection
                                      # criterion at once, and the delta measures
                                      # nothing. Validation targets, masks, HQ
                                      # membership and select_on all follow this.
    target: str = "y_"                # "g_" for the graded ablation
    n_series: int = 6
    n_slices: int = 24                # series mode: slices/series.
                                      # slot mode: 2.5D GROUPS per slot, drawn
                                      # from the 12 cached slices.
    img_size: int = 256
    cache: str = "series"             # series | slot | maxspan. `slot` reads the
                                      # 130mm/336px slot cache; n_series is
                                      # ignored there, slots are fixed at 6.
    band_lo: float = 0.20             # slot mode only: which slot cache to read.
    band_hi: float = 0.80             # 0.06/0.94 is the wider band the 0.935
                                      # notebook uses -- it states the 20-80 band
                                      # was "measurably costing accuracy" on the
                                      # collateral ligaments and lateral meniscus,
                                      # which are two of our three worst labels.
    crop_mm: float = 130.0            # slot mode only: which slot cache to read.
                                      # Recorded in config.json so the bundler
                                      # can tell inference what crop to decode
                                      # at -- a wrong crop changes no shape.
    batch_size: int = 4
    accum: int = 2
    epochs: int = 12
    lr: float = 3e-4
    encoder_lr_mult: float = 0.1      # pretrained encoder wants a gentler LR
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    mixup: str = "manifold"           # manifold | none | pixel
    mixup_alpha: float = 0.3
    ema_decay: float = 0.999
    workers: int = 8
    seed: int = 0
    overfit: int = 0                  # >0: memorise N studies as a pipeline gate
    no_augment: bool = False          # disable SeriesAugment; for baseline runs
    native_nb: bool = False           # slot mode: read the `_nb` cache, whose 2.5D
                                      # side channels are each centre's TRUE source
                                      # neighbours (~3.5 mm) instead of neighbouring
                                      # CACHED planes (~6.5 mm). Centres unchanged.
    cache_slices: int = 12            # slot mode: centre planes per slot in the cache
                                      # (12 = the verified caches). Not n_slices, which
                                      # is the number of 2.5D GROUPS drawn per slot.
    syn_aux: float = 0.0              # experiment B: auxiliary Synovitis supervision on
                                      # report-silent cells, total weight = this fraction
                                      # of the fold's observed Synovitis weight. 0 = off.
    aug_off: str = ""                 # "" = full stack. One of augment.FAMILIES
                                      # to zero that family's probability gates.
    compartment: str = ""             # slot mode: read a half-FOV cache centred on
                                      # one compartment ("medial"/"lateral"), paired
                                      # with a SMALLER crop_mm -- same 336px over half
                                      # the anatomy. "" = the centred full-width cache.
    cache_img: int = 336              # slot mode: pixels per side OF THE CACHE to read.
                                      # Deliberately NOT img_size: a 224 model reading the
                                      # 336 cache (cnxd3lor224) downsamples on purpose, and
                                      # tying cache choice to img_size would break that. To
                                      # test resolution, move BOTH, so the pixels are real.
    encoder_init: str = ""            # path to an SSL-pretrained encoder (artifacts/ssl/*.pt).
                                      # "" = timm's pretrained weights, i.e. every run so far.
    encoder_init_sha: str = ""        # expected sha256[:16] of that encoder's weights. Recorded
                                      # in config.json so a gate can prove WHICH encoder ran.
    exclude_folds: str = ""           # comma-separated folds removed from TRAINING (validation is
                                      # untouched). Needed so a domain-pretrained arm and its control
                                      # train on the SAME restricted studies the SSL encoder saw --
                                      # an unrestricted control is not a matched control.
    run_tag: str = ""                 # suffix on the run dir, so hyperparameter
                                      # variants do not overwrite each other
    freeze_epochs: int = 0            # keep the encoder frozen for N epochs, then
                                      # unfreeze at lr*encoder_lr_mult. Frozen
                                      # DINOv3 scored 0.741 vs 0.655 fine-tuned:
                                      # the head must stabilise before the
                                      # pretrained features are allowed to move.
    labels: str = ""                  # comma-separated subset -> SPECIALIST.
                                      # "" trains all 12. Zeroing w on the rest
                                      # concentrates every shared parameter on
                                      # these labels. Multi-task learning is
                                      # itself a regulariser, so a narrower
                                      # subset is NOT automatically better.
    select_on: str = "tier1"          # tier1 | tier1_hq. Epoch selection stays
                                      # on tier1 by DEFAULT so every existing run
                                      # stays comparable; tier1_hq is reported
                                      # always and is the CROSS-RUN promotion
                                      # metric. See rsna/labels/quality.py.
    loss: str = "bce"                 # bce | rank | both. `rank` is a smooth
                                      # AUC surrogate -- see pairwise_rank_loss.
    rank_w: float = 1.0               # weight on the rank term when loss=both
    rank_tau: float = 1.0             # logit scale inside the pair sigmoid
    rank_trim: float = 0.0            # rpAUC trimming scale. 0 = off. Per-label
                                      # alpha=beta=(1-a)/2 from the MEASURED
                                      # contamination -- see quality.contamination_trim
    rank_trim_warmup: int = 5         # no trimming before this epoch: a fresh
                                      # model's high-loss instances are arbitrary,
                                      # so early trimming deletes random data.
    rank_macro: bool = False          # normalise each label by its OWN pair mass
                                      # then average -- what macro AUC scores.
                                      # Default False ONLY for comparability with
                                      # the fold-0 runs; see the 47.7x spread.
    swa_epochs: int = 5               # average weights over the last N epochs.
                                      # 0 disables. The public MaxSpan checkpoint
                                      # is an SWA over three.
    head_type: str = "hier"           # hier | mil. `mil` replaces the whole
                                      # aggregator with the public 0.924 model's
                                      # flat per-label attention over all S*k
                                      # windows -- no positions, no slot ids.
    n_regions: int = 0                # 0 = off. 4 = 2x2 spatial summaries per
                                      # slice image, appended ONLY to the final
                                      # label-attention memory: 6+24 = 30 tokens
                                      # -> 6+24+96 = 126. Everything upstream is
                                      # untouched, so this does not re-open the
                                      # aggregation experiment. Adds region_emb,
                                      # so a mismatch fails STRICT loading and is
                                      # named explicitly by load_for_inference.
    fine_attn: bool = False           # label queries attend over slot x slice
                                      # tokens, not pre-pooled slot vectors.
                                      # Recorded in config.json because it adds
                                      # NO parameters -- see model.load_for_inference
    
def masked_bce(logits, y, w):
    l = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (l * w).sum() / w.sum().clamp_min(1.0)

def manifold_mixup(o, y, w, alpha, gen):
    """Mix the (B,12,D) penultimate embedding, not the pixels.

    Pixel MixUp is ill-defined here: studies carry different numbers of series
    in different orders, so blending slot i of A with slot i of B can mix a
    sagittal T1 with an axial PD-FS. The embedding is shape-stable.

    Targets mix by EVIDENCE, not by lambda: mixing a masked target against an
    unmasked one with plain lambda-blending would invent a label out of a
    `not_mentioned`, which is the assumption Phase 1 exists to avoid.
    """
    lam = float(np.random.default_rng(gen).beta(alpha, alpha))
    perm = torch.randperm(o.shape[0], device=o.device)
    o = lam * o + (1.0 - lam) * o[perm]
    wa, wb, ya, yb = w, w[perm], y, y[perm]
    w_mix = lam * wa + (1.0 - lam) * wb
    y_mix = (lam * wa * ya + (1.0 - lam) * wb * yb) / w_mix.clamp_min(1e-8)
    return o, torch.where(w_mix > 0, y_mix, torch.zeros_like(y_mix)), w_mix

def load_domain_encoder(encoder, path: str, expect_sha: str = "") -> str:
    """Load an SSL-pretrained encoder BEFORE EMA and optimizer construction.

    ORDER MATTERS. EMA deepcopies the model, so loading afterwards would leave the shadow
    tracking the ORIGINAL DINOv2 weights while the live model used the new ones -- the run
    would then evaluate a mixture and nothing would announce it.
    Strict key/shape checking, plus a content hash, so a run can prove which encoder it used.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    sd = blob["encoder"] if isinstance(blob, dict) and "encoder" in blob else blob
    h = hashlib.sha256(b"".join(sd[k].cpu().numpy().tobytes()
                                for k in sorted(sd))).hexdigest()[:16]
    if expect_sha and h != expect_sha:
        raise SystemExit(f"encoder-init sha mismatch: file {h} != config {expect_sha}")
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"encoder-init key mismatch: missing={list(missing)[:5]} "
                         f"unexpected={list(unexpected)[:5]}")
    meta = blob.get("meta", {}) if isinstance(blob, dict) else {}
    print(f"encoder init: {path} | sha {h} | tag {meta.get('tag','?')} "
          f"| steps {meta.get('steps','?')} | pretrained on {meta.get('folds','?')}")
    return h


class EMA:
    """Exponential moving average of weights; cheap and reliably worth a little."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.state_dict().values(),
                        model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
            else:
                s.copy_(p)
    
def rpauc_keep(logits, y, w, trim):
    """rpAUC instance selection (Xie et al. 2024, Def. 11 + Prop. 12) -> (B,L) mask.

    Delete the `beta` lowest-scoring positives and `alpha` highest-scoring
    negatives per label. Prop. 12 proves those ARE the top-loss instances, since
    the loss is monotone in score -- so no per-instance loss history is needed
    and this is computable inside the batch.

    DEVIATION from Algorithm 1, stated because it matters: the paper trims over
    the WHOLE set before sampling batches; we trim within a batch of 16. Same
    expected fraction, higher variance, and some globally-fine instances get
    dropped on batches where they happen to be the local worst.
    """
    keep = w > 0
    for j in range(logits.shape[1]):
        f = float(trim[j])
        if f <= 0:
            continue
        for pos in (True, False):
            m = keep[:, j] & ((y[:, j] > 0.5) if pos else (y[:, j] <= 0.5))
            n = int(m.sum())
            e = n * f
            k = int(e) + (1 if torch.rand(1).item() < (e - int(e)) else 0)
            if k < 1 or n - k < 1:
                continue
            z = logits[m, j]
            # positives: drop the LOWEST scores. negatives: drop the HIGHEST.
            drop = torch.topk(z, k, largest=not pos).indices
            idx = torch.nonzero(m, as_tuple=True)[0][drop]
            keep[idx, j] = False
    return keep


def pairwise_rank_loss(logits, y, w, tau: float = 1.0, macro: bool = False,
                       trim=None):
    """A differentiable surrogate for the metric we are actually scored on.

    BCE with a soft target t is optimised by sigma(z) -> t, so it spends capacity
    matching MAGNITUDES. AUC ignores magnitudes entirely. That mismatch is not
    hypothetical here: the public labels carry +0.069 more ORDERING information
    than ours (continuous gold58 0.878 vs 0.809) and still trained a WORSE model
    (away_gain -0.0030), because BCE converted the extra grading into a
    calibration target instead of a ranking signal.

    This is the smooth form of `soft_auc`: every pair (i, j) with y_i > y_j
    contributes, weighted by how far apart the targets are.

        sum_{i,j} (y_i - y_j)_+ * softplus(-(z_i - z_j) / tau)
        ------------------------------------------------------
                    sum_{i,j} (y_i - y_j)_+

    The grading survives -- as pair WEIGHT, not as a target value -- so a hedged
    0.6 still outranks a confident 0.95 less strongly than a 0.05 does, while the
    absolute numbers are discarded.

    Needs pairs, so it wants a wide batch. DINOv2 has no BatchNorm, so
    `--batch-size 16 --accum 1` gives gradients identical to the baseline's
    8 x 2 while doubling the studies a pair term can see.
    """
    B, L = logits.shape
    if trim is not None:
        # rpAUC: contaminated instances are removed BEFORE the pairs are formed,
        # so a deleted study contributes to no pair at all.
        w = w * rpauc_keep(logits.detach(), y, w, trim).float()
    d = logits.unsqueeze(1) - logits.unsqueeze(0)            # (B,B,L) z_i - z_j
    gap = y.unsqueeze(1) - y.unsqueeze(0)                    # (B,B,L) y_i - y_j
    pw = torch.clamp(gap, min=0.0) * (w.unsqueeze(1) * w.unsqueeze(0))
    per = F.softplus(-d / tau)
    if not macro:
        # GLOBAL normalisation. Measured pair-weight share across the corpus spans
        # **47.7x**: Synovitis gets 0.36% and Effusion 17.40%, against the 8.33%
        # macro AUC would give each. Pair weight is sum (y_i - y_j)+, so a hedged
        # target near 0.5 contributes almost nothing -- and hedged targets are
        # exactly the low-label_q ones. Kept as the default only so the fold-0
        # runs stay comparable; it is NOT the right objective.
        tot = pw.sum()
        return logits.sum() * 0.0 if tot <= 0 else (pw * per).sum() / tot
    # MACRO normalisation: normalise each label by its OWN pair mass, then average
    # the labels equally -- which is what macro AUC actually scores.
    num, den = (pw * per).sum(dim=(0, 1)), pw.sum(dim=(0, 1))
    ok = den > 0
    return (num[ok] / den[ok]).mean() if ok.any() else logits.sum() * 0.0


def soft_auc(y_soft, score, valid):
    """AUC generalised to GRADED targets, per label, averaged.

    Tier 1 binarises at 0.5. That is fine for our own labels (8.2% fractional)
    and destroys the public ones (99.9% fractional): measured against gold58 the
    public labels score 0.878 continuous and 0.782 once thresholded, i.e. their
    entire +0.069 advantage lives in the fractional part. A metric that throws it
    away cannot see the change we are making.

    This is the concordance every pair (i,j) with y_i > y_j contributes, weighted
    by how far apart the targets are:

        sum_{y_i > y_j} (y_i - y_j) * [s_i > s_j] + 0.5 * [s_i == s_j]
        ------------------------------------------------------------
                        sum_{y_i > y_j} (y_i - y_j)

    Reduces EXACTLY to AUC when y is 0/1, so the two are on one scale.
    """
    per = {}
    for j, l in enumerate(LABELS):
        m = valid[:, j]
        if m.sum() < 8:
            continue
        y, s = y_soft[m, j], score[m, j]
        d = y[:, None] - y[None, :]           # (n,n) target gap
        w = np.clip(d, 0, None)               # only pairs where i outranks j
        tot = w.sum()
        if tot <= 0:
            continue
        c = (s[:, None] > s[None, :]).astype(np.float32)
        c += 0.5 * (s[:, None] == s[None, :])
        per[l] = float((w * c).sum() / tot)
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


def macro_auc(y_true, score, valid):
    """Per-label AUC over valid entries; labels with one class are skipped."""
    per = {}
    for j, l in enumerate(LABELS):
        m = valid[:, j]
        if m.sum() < 8:
            continue
        t = y_true[m, j]
        if len(np.unique(t)) < 2:
            continue
        per[l] = roc_auc_score(t, score[m, j])
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


@torch.no_grad()
def evaluate(model, loader, device, keep=None, hq=None):
    model.eval()
    P, Y, W, G = [], [], [], []
    for b in loader:
        logits = model(b["img"].to(device, non_blocking=True),
                       b["series_type"].to(device), b["slice_pos"].to(device),
                       b["series_mask"].to(device))
        P.append(logits.float().cpu().numpy())
        Y.append(b["y"].numpy()); W.append(b["w"].numpy()); G.append(b["gold"].numpy())
    P, Y, W, G = map(np.concatenate, (P, Y, W, G))

    # Tier 1: agreement with the LLM labels -- biased but n~880, SE~0.02
    t1, per1 = macro_auc((Y > 0.5).astype(int), P, W > 0)
    # Tier 2: the real target -- unbiased but n~12/fold, SE~0.06. VETO ONLY.
    # Scores ALL 12 labels including the ambiguous-laterality ones, because the
    # hidden test set does; masking them here would measure a different task.
    gm = ~np.isnan(G)
    if keep is not None:
        # Otherwise T2 averages 12 labels a specialist never trained on, and
        # reports a number that describes a different model than the one running.
        gm = gm & keep[None, :]
    t2, per2 = macro_auc(np.nan_to_num(G).astype(int), P, gm)
    # Tier 1 HQ: the same n~880 studies, but only the labels whose TARGET is
    # trustworthy (label_q >= 0.85 vs gold). Tier 1 spends ~17% of its weight on
    # Synovitis (0.676) and Effusion (0.696), where a model can move the number
    # without moving true skill. Reported alongside, never instead of -- every
    # run we already have is recorded on the twelve-label version.
    t1_hq = float("nan")
    if hq:
        v = [a for l, a in per1.items() if l in hq]
        t1_hq = float(np.mean(v)) if v else float("nan")
    # Tier 1 SOFT: the same studies and cells, but the graded target intact.
    # Required to compare label policies at all -- see soft_auc.
    t1s, per1s = soft_auc(Y, P, W > 0)
    # Validation BCE against the soft target. NOT comparable across policies
    # (each has its own irreducible entropy) but the cleanest within-policy
    # early-stopping signal, and it is what the loss actually optimises.
    q = 1.0 / (1.0 + np.exp(-np.clip(P, -30, 30)))
    m = W > 0
    bce = float(-(Y[m] * np.log(q[m] + 1e-7)
                  + (1 - Y[m]) * np.log(1 - q[m] + 1e-7)).mean())
    return dict(tier1=t1, tier1_hq=t1_hq, tier1_soft=t1s, val_bce=bce, tier2=t2,
                per_label_tier1=per1, per_label_tier1_soft=per1s,
                per_label_tier2=per2, n_gold=int(gm.any(axis=1).sum()))
    
def make_loaders(cfg):
    lab = pd.read_parquet(ROOT / f"artifacts/labels/labels_llm_{cfg.policy}.parquet")
    # The validation frame is a SEPARATE object whenever val_policy is set, so a
    # policy experiment moves the training target only. Same studies, same folds,
    # same masks-from-the-ruler; only y_ differs between the two frames.
    vpol = cfg.val_policy or cfg.policy
    lab_va = (lab if vpol == cfg.policy else
              pd.read_parquet(ROOT / f"artifacts/labels/labels_llm_{vpol}.parquet"))
    folds = pd.read_parquet(ROOT / "artifacts/folds.parquet")
    meta = pd.read_parquet(ROOT / "artifacts/series_meta.parquet")
    meta = meta[meta.error.isna()].merge(
        pd.read_csv(ROOT / "data/train_series.csv")[["SeriesInstanceUID",
                                                     "Fluid_Sensitive"]],
        on="SeriesInstanceUID", how="left")
    meta["Fluid_Sensitive"] = meta.Fluid_Sensitive.fillna(0).astype(int)

    tr = folds.index[folds.fold != cfg.fold].tolist()
    va = folds.index[folds.fold == cfg.fold].tolist()
    if cfg.exclude_folds:
        drop = {int(x) for x in cfg.exclude_folds.split(",") if x.strip()}
        assert cfg.fold not in drop, "cannot exclude the validation fold from training"
        keep = set(folds.index[~folds.fold.isin(drop)])
        n0 = len(tr); tr = [s for s in tr if s in keep]
        print(f"exclude-folds {sorted(drop)}: training studies {n0} -> {len(tr)}; "
              f"validation fold {cfg.fold} untouched ({len(va)} studies)")
    # config_without("") IS AugConfig(), so the control and the ablation arms
    # are constructed through one code path.
    aug = SeriesAugment(config_without(cfg.aug_off)); assert_no_hflip(aug)
    if cfg.aug_off:
        from rsna.train.augment import FAMILIES
        print(f"augmentation: family {cfg.aug_off!r} OFF "
              f"({', '.join(FAMILIES[cfg.aug_off])} = 0)", flush=True)
    if cfg.no_augment:
        aug = None          # establish that the model learns before regularising it
    if cfg.overfit:
        # Gate, not an experiment: train and validate on the SAME few studies
        # with no augmentation and no mixup. A pipeline that cannot memorise 64
        # studies has a defect -- misaligned labels, a dead gradient, a mask
        # that zeroes everything -- and no amount of epochs will reveal it.
        tr = tr[:cfg.overfit]; va = list(tr); aug = None
    if cfg.cache_slices != 12 and cfg.cache != "slot":
        raise SystemExit(f"--cache-slices requires --cache slot (got {cfg.cache!r})")
    if cfg.native_nb and cfg.cache != "slot":
        # Only the slot cache has an _nb variant. Silently ignoring the flag
        # would train a default-cache model under a name claiming otherwise.
        raise SystemExit(f"--native-nb requires --cache slot (got {cfg.cache!r})")
    if cfg.cache == "maxspan":
        # A THIRD cache -- the public 0.924 notebook's representation, reproduced
        # verbatim (rsna/data/maxspan.py). Same tensor contract again, so the
        # model and collate are untouched and a delta isolates the pixels.
        cdir = ROOT / "cache_maxspan_336_140mm_64sl/train"
        if not (cdir.parent / "manifest_train.json").is_file():
            raise SystemExit(f"{cdir} incomplete -- run:\n"
                             f"  python -m rsna.data.build_maxspan_cache "
                             f"--split train --workers 24")
        common = dict(labels=lab, folds=folds, cache_dir=cdir,
                      n_groups=cfg.n_slices, img_size=cfg.img_size,
                      target_prefix=cfg.target,
                      label_subset=[x.strip() for x in cfg.labels.split(",")
                                    if x.strip()] or None)
        DS = MaxSpanDataset
    elif cfg.cache == "slot":
        # A second, independent cache -- see rsna/data/build_slot_cache.py. The
        # tensor contract is identical, so KneeModel and collate_studies are
        # unchanged; only where the pixels come from differs.
        from rsna.data.build_slot_cache import cache_dir as _slot_dir
        band = (cfg.band_lo, cfg.band_hi)
        cdir = _slot_dir(cfg.crop_mm, band, cfg.native_nb, cfg.cache_slices,
                         cfg.compartment, cfg.cache_img) / "train"
        if not cdir.is_dir():
            raise SystemExit(f"{cdir} missing -- run:\n"
                             f"  python -m rsna.data.build_slot_cache "
                             f"--crop-mm {cfg.crop_mm:.0f} "
                             f"--band {cfg.band_lo} {cfg.band_hi}"
                             f"{' --native-nb' if cfg.native_nb else ''}"
                             f" --n-slices {cfg.cache_slices}"
                             f"{f' --compartment {cfg.compartment}' if cfg.compartment else ''}"
                             f"{f' --img {cfg.cache_img}' if cfg.cache_img != 336 else ''}")
        common = dict(labels=lab, folds=folds, cache_dir=cdir,
                      n_groups=cfg.n_slices, img_size=cfg.img_size,
                      target_prefix=cfg.target,
                      label_subset=[x.strip() for x in cfg.labels.split(",")
                                    if x.strip()] or None)
        DS = SlotDataset
    else:
        common = dict(labels=lab, series_meta=meta, folds=folds,
                      cache_dir=ROOT / "cache", n_series=cfg.n_series,
                      n_slices=cfg.n_slices, img_size=cfg.img_size,
                      target_prefix=cfg.target)
        DS = StudyDataset
    # overfit gate uses DETERMINISTIC sampling on the train split too: with
    # train=True the slice jitter and series shuffle resample every epoch, so
    # the model never sees the same input twice and memorisation is impossible.
    aux_kw = {}
    if cfg.syn_aux > 0:
        # Experiment B (registered 2026-09-15). Eligibility and targets are PRECOMPUTED
        # (artifacts/labels/syn_aux_v1eff.parquet), never derived from gold. The weight
        # scale `a` is computed from THIS fold's TRAINING studies only, so the auxiliary
        # coefficient mass is exactly syn_aux x the observed Synovitis weight.
        if cfg.cache != "slot":
            raise SystemExit("--syn-aux requires --cache slot")
        if cfg.mixup != "none":
            raise SystemExit("--syn-aux requires --mixup none (mixup would blend aux targets)")
        at = pd.read_parquet(ROOT / "artifacts/labels/syn_aux_v1eff.parquet")
        at = at if at.index.name == "StudyInstanceUID" else at.set_index("StudyInstanceUID")
        e = at.loc[[u for u in tr if u in at.index]]
        w_syn = float(lab.loc[tr, "w_Synovitis"].sum())
        elig_w = float((e.elig.astype(float) * e.w_eff).sum())
        a_scale = cfg.syn_aux * w_syn / elig_w
        el = e[e.elig]
        aux_kw = {"aux": {u: (float(q), float(a_scale * we))
                          for u, q, we in zip(el.index, el.q_syn, el.w_eff)}}
        print(f"syn_aux: {len(el)} eligible train studies | observed Synovitis weight "
              f"{w_syn:.1f} | eligible Effusion weight {elig_w:.1f} | a = {a_scale:.5f} "
              f"| aux mass {a_scale*elig_w:.1f} = {cfg.syn_aux:.0%} of observed", flush=True)
    ds_tr = DS(tr, train=not cfg.overfit, transform=aug, seed=cfg.seed, **common, **aux_kw)
    ds_va = DS(va, train=False, transform=None, seed=cfg.seed,
               **{**common, "labels": lab_va})
    mk = lambda d, sh: DataLoader(d, batch_size=cfg.batch_size, shuffle=sh,
                                  num_workers=cfg.workers, pin_memory=True,
                                  collate_fn=collate_studies, drop_last=sh,
                                  persistent_workers=cfg.workers > 0)
    return mk(ds_tr, True), mk(ds_va, False), len(tr), len(va)

def run(cfg: Cfg):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = "cuda"
    # A real-ALLOCATION gate is not enough. This H100 periodically wedges into a
    # state where nvidia-smi reports it fine, torch.cuda.is_available() returns
    # True, and allocation SUCCEEDS -- but every kernel launch fails with
    # `unspecified launch failure`. That killed a fold ten minutes in and left a
    # corrupt checkpoint. Allocation and launch are different failure surfaces,
    # so the gate has to actually COMPUTE.
    x = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    for _ in range(20):
        x = x @ x.T / 1000.0
    torch.cuda.synchronize()
    if not torch.isfinite(x).all():
        raise SystemExit("GPU gate: sustained compute produced non-finite values")
    del x
    torch.cuda.empty_cache()

    tr, va, n_tr, n_va = make_loaders(cfg)
    # Which labels weak-OOF may be steered on. Computed from THIS run's label
    # policy, because label_q is a property of the target file, not of the model.
    # From the RULER's policy, not the training policy. label_q is a property of
    # the target file being scored against, and holding the ruler fixed is the
    # whole point of val_policy -- reading it from cfg.policy would let a policy
    # experiment silently redefine T1hq and the promotion criterion under itself.
    hq_set = {l for l, k in zip(LABELS,
                                high_quality(cfg.val_policy or cfg.policy)) if k}
    sel_key = cfg.select_on          # NOT `sel` -- taken below by --labels
    if sel_key not in ("tier1", "tier1_hq", "tier1_soft"):
        raise SystemExit(f"--select-on must be tier1|tier1_hq|tier1_soft, got {sel_key!r}")
    # FLAT rate on every label. The per-label table derived from gold58 is
    # RETRACTED: it used Youden's J, while the paper's coefficient is
    # P(Y=1|W=1) - P(Y=1|W=0) -- prevalence-blind vs not. MCL reads 0.805 under J
    # and 0.529 under the paper, with 47.1% of its weak positives gold-negative,
    # so the table trimmed nearly the wrong labels. And the observed-negative
    # counts behind those estimates are 1 to 13 studies. See quality.py.
    TRIM = None
    if cfg.rank_trim > 0:
        TRIM = torch.full((len(LABELS),), float(cfg.rank_trim), device=device)
        print(f"rpAUC: flat {cfg.rank_trim:.0%} trim per side, "
              f"warm-up {cfg.rank_trim_warmup} epochs, rank term only")
    bests: dict[str, float] = {}
    swa, swa_n = None, 0
    print(f"promotion metric: T1hq over {len(hq_set)} labels "
          f"(label_q >= {HQ_MIN}) | epoch selection on {sel_key}")
    model = KneeModel(cfg.backbone, img_size=cfg.img_size, fine_attn=cfg.fine_attn,
                      head_type=cfg.head_type, n_regions=cfg.n_regions).to(device)
    # Log the RESOLVED norm layout, not the requested one. timm's ViT names its
    # final LayerNorm `norm` or `fc_norm` depending on a fallback that fires only
    # when PRETRAINED weights fail to load -- so the layout is a property of this
    # process, not of the config, and it is what the regional branch's
    # `encoder.fc_norm(r)` routes through. test_regions.py prints it for the model
    # IT builds; that tells us nothing about the one actually being trained here.
    _e = model.encoder
    print(f"model: n_regions={cfg.n_regions} head={cfg.head_type} "
          f"fine_attn={cfg.fine_attn} | encoder norm={type(getattr(_e,'norm',None)).__name__} "
          f"fc_norm={type(getattr(_e,'fc_norm',None)).__name__} "
          f"prefix_tokens={getattr(_e,'num_prefix_tokens','n/a')}")
    if cfg.encoder_init:                      # BEFORE EMA: the shadow must copy these weights
        load_domain_encoder(model.encoder, cfg.encoder_init, cfg.encoder_init_sha)
    ema = EMA(model, cfg.ema_decay)

    enc = {"params": list(model.encoder.parameters()),
           "lr": cfg.lr * cfg.encoder_lr_mult}
    rest = {"params": [p for n, p in model.named_parameters()
                       if not n.startswith("encoder.")], "lr": cfg.lr}
    opt = torch.optim.AdamW([enc, rest], weight_decay=cfg.weight_decay)
    steps = max(1, len(tr) // cfg.accum) * cfg.epochs
    warm = int(steps * cfg.warmup_frac)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        (s + 1) / max(warm, 1) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(steps - warm, 1)))))

    # FULL backbone name, not split('.')[0]: convnext_tiny.fb_in1k and
    # convnext_tiny.dinov3_lvd1689m are different models and must not share a
    # directory -- the second silently overwrote the first's checkpoint.
    safe = cfg.backbone.replace(".", "_").replace("/", "_")
    if cfg.cache != "series":
        safe = f"{safe}__{cfg.cache}"   # a slot run and a series run of the same
                                        # backbone are different models
        if cfg.crop_mm != 130.0:        # ...and so are two crops of one cache
            safe = f"{safe}{cfg.crop_mm:.0f}"
        if (cfg.band_lo, cfg.band_hi) != (0.20, 0.80):   # ...and two bands
            safe = f"{safe}b{int(cfg.band_lo*100):02d}{int(cfg.band_hi*100):02d}"
        if cfg.cache_slices != 12:      # ...and two centre-plane counts
            safe = f"{safe}s{cfg.cache_slices}"
        if cfg.native_nb:               # ...and two ways of filling the channels
            safe = f"{safe}nb"
        if cfg.syn_aux > 0:             # ...and a model with auxiliary supervision
            safe = f"{safe}sa{int(round(cfg.syn_aux * 100))}"
        if cfg.compartment:             # ...and two halves of the same joint
            safe = f"{safe}{cfg.compartment[:3]}"
        if cfg.cache_img != 336:        # ...and two resolutions of the same pixels
            safe = f"{safe}i{cfg.cache_img}"
        if cfg.encoder_init:            # ...and two initialisations of the same architecture
            safe = f"{safe}dp"
        if cfg.exclude_folds:           # ...and two training populations
            safe = f"{safe}x{cfg.exclude_folds.replace(',', '')}"
    if cfg.run_tag:
        safe = f"{safe}__{cfg.run_tag}"
    if cfg.fine_attn:
        safe = f"{safe}_fine"
    sel = [x.strip() for x in cfg.labels.split(",") if x.strip()]
    if sel:
        missing = [x for x in sel if x not in LABELS]
        if missing:
            raise SystemExit(f"--labels names unknown findings: {missing}\n"
                             f"valid: {LABELS}")
        safe = f"{safe}_spec{len(sel)}"
    keep_lbl = np.array([l in sel for l in LABELS]) if sel else None
    out = ROOT / f"artifacts/runs/f{cfg.fold}_{safe}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(f"train {n_tr} / val {n_va} studies | {len(tr)} steps/epoch -> {out}")

    best, hist, step = -1.0, [], 0
    # tqdm on stderr (its default) so stdout keeps only the epoch summaries.
    # mininterval=2 keeps the log readable when the run is tee'd to a file.
    ebar = tqdm(range(cfg.epochs), desc="epochs", unit="ep", mininterval=2,
                dynamic_ncols=True, position=0)
    for ep in ebar:
        model.train(); t0, run_loss = time.time(), 0.0
        frozen = ep < cfg.freeze_epochs
        model.set_encoder_trainable(not frozen)
        if frozen:
            # eval() too, not just requires_grad=False: DropPath and dropout
            # inside the encoder would otherwise keep perturbing "frozen" features
            model.encoder.eval()
            if ep == 0:
                print(f"encoder FROZEN for epochs 0-{cfg.freeze_epochs-1}")
        elif ep == cfg.freeze_epochs and cfg.freeze_epochs:
            print(f"encoder UNFROZEN at epoch {ep}, lr {cfg.lr*cfg.encoder_lr_mult:.1e}")
        bbar = tqdm(tr, desc=f"ep{ep:02d}", unit="b", leave=False,
                    mininterval=2, dynamic_ncols=True, position=1)
        for i, b in enumerate(bbar):
            img = b["img"].to(device, non_blocking=True)
            y, w = b["y"].to(device), b["w"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model.forward_embed(img, b["series_type"].to(device),
                                        b["slice_pos"].to(device),
                                        b["series_mask"].to(device))
                if cfg.mixup == "manifold":
                    o, y, w = manifold_mixup(o, y, w, cfg.mixup_alpha, cfg.seed + step)
                logits = model.head(o)
                # BCE and the rank term are computed on the SAME logits. `rank`
                # is scale-free (only differences matter), so a model trained on
                # it alone has arbitrary calibration -- fine for AUC, and the
                # reason `both` exists for anyone who wants a usable sigmoid.
                if cfg.loss == "bce":
                    loss = masked_bce(logits, y, w)
                elif cfg.loss == "rank":
                    loss = pairwise_rank_loss(logits, y, w, cfg.rank_tau, cfg.rank_macro,
                                       TRIM if ep >= cfg.rank_trim_warmup else None)
                elif cfg.loss == "both":
                    loss = (masked_bce(logits, y, w)
                            + cfg.rank_w * pairwise_rank_loss(logits, y, w, cfg.rank_tau, cfg.rank_macro,
                                       TRIM if ep >= cfg.rank_trim_warmup else None))
                else:
                    raise SystemExit(f"--loss must be bce|rank|both, got {cfg.loss!r}")
                if cfg.syn_aux > 0:
                    # SEPARATE term, ORIGINAL normaliser: the old BCE and rank terms are
                    # numerically unchanged, and the aux cells add coefficient mass
                    # without entering the denominator or the rank pairs.
                    ya, wa = b["y_aux"].to(device), b["w_aux"].to(device)
                    la = F.binary_cross_entropy_with_logits(logits, ya, reduction="none")
                    loss = loss + (la * wa).sum() / w.sum().clamp_min(1.0)
                loss = loss / cfg.accum
            loss.backward()
            run_loss += loss.item() * cfg.accum
            if (i + 1) % cfg.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); sched.step()
                ema.update(model); step += 1
            if i % 20 == 0:
                # base rate is ~0.596; a run stuck there is not learning
                bbar.set_postfix(loss=f"{run_loss/(i+1):.4f}",
                                 lr=f"{sched.get_last_lr()[-1]:.2e}", refresh=False)
        bbar.close()

        m_raw = evaluate(model, va, device, keep_lbl, hq_set)
        m_ema = evaluate(ema.shadow, va, device, keep_lbl, hq_set)
        m = m_ema if m_ema[sel_key] > m_raw[sel_key] else m_raw
        which = "ema" if m is m_ema else "raw"
        hist.append(dict(epoch=ep, loss=run_loss / max(len(tr), 1),
                         secs=time.time() - t0, which=which, **{
                             k: v for k, v in m.items() if not k.startswith("per_")}))
        print(f"ep{ep:02d} loss {hist[-1]['loss']:.4f} | T1 {m['tier1']:.4f} "
              f"| T1hq {m['tier1_hq']:.4f} | T1s {m['tier1_soft']:.4f} "
              f"| bce {m['val_bce']:.4f} | T2 {m['tier2']:.4f} (n={m['n_gold']}) "
              f"| {which} | {hist[-1]['secs']:.0f}s")

        ebar.set_postfix(T1=f"{m['tier1']:.4f}", T1hq=f"{m['tier1_hq']:.4f}",
                         T2=f"{m['tier2']:.4f}", loss=f"{hist[-1]['loss']:.4f}",
                         best=f"{max(best, m[sel_key]):.4f}")
        # MULTI-CRITERION checkpoints. `best.pt` stays exactly what it was --
        # every bundle and notebook resolves that name -- and the alternatives
        # are saved alongside so a promotion decision made on one metric does not
        # force us to retrain to test another.
        picked = ema.shadow if which == "ema" else model
        for key, crit, better in (("best", sel_key, max), ("best_tier1", "tier1", max),
                                  ("best_tier1_hq", "tier1_hq", max),
                                  ("best_tier1_soft", "tier1_soft", max),
                                  ("best_val_bce", "val_bce", min)):
            cur = m[crit]
            if not np.isfinite(cur):
                continue
            prev = bests.get(key)
            if prev is None or better(cur, prev) == cur and cur != prev:
                bests[key] = cur
                torch.save({k: v.cpu() for k, v in picked.state_dict().items()},
                           out / f"{key}.pt")
                (out / f"{key}_metrics.json").write_text(json.dumps(m, indent=2))
                if key == "best":
                    best = cur
        # SWA over the LAST swa_epochs. Averaging weights, not predictions, so it
        # costs one extra state dict and no extra forward passes. The public
        # MaxSpan checkpoint is itself an SWA over three epochs.
        if cfg.swa_epochs and ep >= cfg.epochs - cfg.swa_epochs:
            sd = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
            if swa is None:
                swa, swa_n = sd, 1
            else:
                swa_n += 1
                for k in swa:
                    swa[k] += (sd[k] - swa[k]) / swa_n
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / "last.pt")
    if swa is not None:
        # NOTE: any BatchNorm running stats are averaged, not recomputed. Correct
        # for LayerNorm backbones (DINOv2); for CoAtNet/ConvNeXt a BN-recalibration
        # pass over the train loader would be the rigorous version. Flagged, not
        # silently assumed away.
        torch.save({k: v.cpu() for k, v in swa.items()}, out / f"swa{swa_n}.pt")
        print(f"SWA over the last {swa_n} epochs -> {out}/swa{swa_n}.pt")
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    print("checkpoints: " + ", ".join(f"{k}={v:.4f}" for k, v in bests.items()))
    print(f"best {sel_key} {best:.4f} -> {out}/best.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    def _bool(x):
        # type=bool would make "--flag False" evaluate to True, since
        # bool("False") is True. Parse the string properly instead.
        return str(x).lower() in ("1", "true", "yes", "y")
    for k, v in asdict(Cfg()).items():
        p.add_argument(f"--{k.replace('_','-')}",
                       type=_bool if isinstance(v, bool) else type(v),
                       nargs="?" if isinstance(v, bool) else None,
                       const=True if isinstance(v, bool) else None, default=v)
    a = p.parse_args()
    run(Cfg(**{k.replace('-','_'): v for k, v in vars(a).items()}))