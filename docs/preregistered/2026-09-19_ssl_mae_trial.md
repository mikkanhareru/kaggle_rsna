# S2: bounded MAE trial — continued in-domain pretraining of DINOv2 ViT-S/14

Registered 2026-09-19, BEFORE the trial runs. Plan: DOMAIN_PRETRAINING_PLAN_V2_2026-09-19.md.
This registration fixes the SSL recipe. The DOWNSTREAM gate (S3) is registered separately, before
any fine-tuned score exists.

## What S0/S1 established (measured, not assumed)
- Corpus: all planes of the six slot series the classifier already selects, **folds 0-3 only**.
  3,525 studies / 16,126 series / **542,591 planes** / 57.06 GiB, 0 build errors.
  **Genuinely new exposure 39.7%** (planes that are neither a cached centre nor a +-1 neighbour) --
  the earlier "3.4x / 819k" framing was wrong and is withdrawn.
- **Overlap parity: 8,268 centre planes across 150 random studies are BIT-IDENTICAL to the supervised
  cache** (0 different). Achieved by reusing each series' existing percentile window rather than
  recomputing it over all planes. SSL pixels ARE downstream pixels where they overlap.
- **Holdout enforced in the builder**: it refuses fold-4 studies, so the cache cannot contain them.
- Throughput, page cache warm (57 GiB pre-read in 126 s; machine has 1,007 GB RAM so the corpus is
  fully resident in steady state): batch 256 -> **1,817 img/s, 14.6 GiB allocated / 15.1 reserved**;
  batch 512 -> 1,948 img/s, 28.8 GiB (+7% for 2x memory). COLD was 253 img/s: the 7x gap is I/O, so
  the launcher pre-reads the cache.
- Pre-flight on the downstream load path: an exported UNMODIFIED encoder reloads to parameters
  identical to the pretrained baseline; the EMA shadow copies the loaded weights (load happens BEFORE
  EMA construction); a wrong content hash is rejected.

## Frozen SSL recipe (no sweep; changing any of these makes it a different experiment)
    objective     MAE, mask ratio 0.75, norm_pix targets, loss on masked patches only
    init          timm vit_small_patch14_dinov2.lvd142m (pretrained), img 336, patch 14
    input         2.5D triplet (c-1, c, c+1) from ONE series, clamped at the ends, ImageNet norm
    sampling      study -> series -> plane, each UNIFORM (uniform-over-planes would over-weight
                  long series and multi-series studies)
    augmentation  random crop scale 0.8-1.0 only. NO flips -- the cache is canonicalised and the
                  deterministic mirror must not be undone.
    batch         256, accum 1        (1,817 img/s; 14.6 GiB allocated)
    optimiser     AdamW, betas (0.9, 0.95), weight decay 0.05, grad clip 1.0, bf16 autocast
    lr            encoder 5e-5, decoder x10 (5e-4), 6,000-step warmup then cosine to 0
    horizon       **300,000 steps** = 76.8M images = ~142 passes over the corpus, ~11.7 GPU-h
    checkpoints   every 50,000 steps, encoder-only + content hash
The horizon is a RESOURCE choice inside plan v2's 12-24 GPU-h cap, not a claim that MAE converges
there. The encoder LR is a judgement (low enough to limit drift from a 142M-image prior, high enough
to adapt); no measurement selects it, and a failure closes THIS recipe, not domain pretraining.

## What is deliberately NOT decided here
The downstream test, its bar and its controls are registered separately before any fine-tuned number
exists. Fixed already by plan v2 and repeated so it cannot drift: the first downstream arm is
**ep50nb** (its native-neighbour inputs match the SSL input), evaluation uses the fold-4 holdout, and
inner-split controls must be RETRAINED on the same restricted studies -- existing fold-0 numbers were
trained with fold 4 included and are not matched controls.

## Diagnostics during the run (not gates)
Reconstruction loss is a reconstruction metric; it does not measure transfer. At most TWO frozen probes
total, and the continuation decision rests on matched fine-tuning (MAE paper sec 4.3: frozen and
fine-tuned trends can differ and reorder methods; and our probe costs 1.46 h/fold against 1.99/1.83 for
fine-tuning, so it is not nearly free).

## Abort
- Loss not decreasing, or NaN/divergence -> stop, record, keep H4.
- Wall clock exceeding the cap -> stop at the cap; the cosine horizon is fixed so a truncated run is
  reported as truncated, not as a completed schedule.
- Nothing promoted by Oct 14 -> ship H4 (LB 0.907).

## RESULT (2026-09-20) -- RUN COMPLETED BUT BROKE AT 55%; three clean checkpoints survive
Timeline: loss fell 0.81 -> 0.289 and was FLAT for ~30k steps, then went NaN suddenly between step
163,450 (loss 0.2894) and 164,300, and never recovered. The remaining 136k steps trained on NaN.
    USABLE   s50000 loss 0.3045 | s100000 0.2941 | s150000 0.2892   (0 NaN tensors)
    CORRUPT  s200000, s250000, s300000, final -- 174 NaN tensors each (identical sha = all-NaN)
MECHANISM (the propagation is certain, the trigger is not): a single non-finite loss or gradient makes
`clip_grad_norm_` compute a NaN scale, which multiplies EVERY parameter -- so the whole model goes NaN
in one step, and AdamW's moments follow. That matches the signature: all 174 tensors NaN at once, not a
localised overflow, and no upward drift in the loss beforehand. The exact triggering batch is not
identifiable from the log; nothing was recorded to identify it, which is itself the bug.
FIX (in rsna/train/pretrain_mae.py, for any future run): skip a batch whose LOSS is non-finite before it
reaches backward; recompute and check the grad norm BEFORE the optimizer step; abort after 20 such
batches. This converts a silent 10-hour loss into a logged skip.

## THE MORE IMPORTANT FINDING: the encoder drifted far more than intended
Distance from the DINOv2 initialisation (relative L2 over all matching tensors):
    s50000  15.9%   s100000  25.2%   s150000  31.1%
    largest-moving tensor at s150000: blocks.11.attn.qkv.weight  **175.8%** (more than rewritten)
For scale: SUPERVISED fine-tuning moves this encoder ~0.4% and that drift is worth 0.066 T1 (ep50frz).
So MAE at encoder lr 5e-5 moved it roughly 78x further than fine-tuning does. **My registered
description of 5e-5 as "low enough to limit drift from a 142M-image prior" was WRONG**, and the
measurement says so: this was not gentle continued pretraining, it substantially reshaped the encoder
toward pixel reconstruction. That is precisely the degraded-transfer risk the review named.
CONSEQUENCE for S3's prior: a 31%-drifted encoder may well be WORSE downstream than plain DINOv2. The
registered decision rule does not change -- matched fine-tuning decides -- but the expectation should be
stated honestly before the number exists, not after.
CHECKPOINT CHOICE, recorded before any downstream score: S3 tests **s150000**, chosen as the last
NON-CORRUPT checkpoint (a corruption/availability criterion), NOT by any downstream score. If it fails,
a lower-LR re-run is a NEW experiment with its own registration -- not a search across the surviving
checkpoints.
