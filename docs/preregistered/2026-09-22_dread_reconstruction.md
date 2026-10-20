# R: reconstruct Dread Development's 0.924 single model, trained by us, judged on the leaderboard

Registered 2026-09-22 ~10:40 JST, BEFORE the corpus exists and before any training.
This is the "untested BUNDLE" recorded in memory (rsna-maxspan-view): every ingredient of the public
0.92-0.94 recipes was null or negative when tested ONE AT A TIME inside our pipeline, and our own notes
say the bundle "only the LB can judge at the ~0.02 scale". This tests the bundle, whole, on the LB.

## Correction recorded first
In chat on 2026-09-22 I merged two notebooks: I said the public 0.935 is "carried by a single CoAtNet
arm recorded at 0.924". Wrong. **0.924 is Dread Development's SINGLE-MODEL notebook** (public LB, one
arm, no ensembling, no TTA). The 0.935 ensemble's CoAtNet arm is recorded at 0.9213 on GOLD58, which is
not an LB score. The conclusion that motivated this survives: a public single model scores 0.924 on the
LB, above our entire 5-member H5 (0.910).

## Reference, pinned to one version (per PUBLIC_RECIPE_REPRODUCTION_STATUS_2026-09-12.md)
Inference: dreaddevelopment/knee-mri-twelve-findings-from-a-single-model -- source pulled 2026-09-22,
code only, NO weights or datasets downloaded; stored in competition_details/reference/ with SHA256SUMS.
It mounts one dataset (raptor-knee-widedense) and runs ONE arm: raptor_ft_coatnet_v4_full.pt.
Training: knee_mri_training_the_twelve_finding_model.py (self-identified training half of this 0.924).

## The audit -- every component marked VERIFIED / DIFFERENT / UNAVAILABLE
    component         0.924 recipe                                           status       what we use
    backbone          coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k @384            VERIFIED     same
    encoder init      "raptor" SSL, their own weights                        UNAVAILABLE  timm ImageNet-12k->1k
    head              RaptorClassifier, per-finding attention-MIL            VERIFIED     their code; train==infer
    geometry          5 slots Sag-f18/Sag-nf14/Cor-f12/Cor8/Ax12 = 64,       VERIFIED     their build_study VERBATIM
                      first-matching series in CSV order, 6-94% span,
                      2-98 pct window per series, 140 mm -> 336 INTER_AREA
    2.5D              3 neighbouring slices -> 3 channels, 336 -> 384 bilin  VERIFIED     same
    sampler           k=12 random centres (train) / evenly spaced (eval),    VERIFIED     their code, quirk KEPT
                      between FIRST and LAST valid slice of the whole stack
    labels            author LLM-read reports -> 12 soft probs, dense        UNAVAILABLE  labels_llm_public y_ values
                      (default file labels_llm_soft.parquet, never shared)   (FORMAT       -- dense, soft on 93.7% of
                                                                             matches)     cells, defined on 100%
    loss              BCE, pos_weight clip((1-p)/p,1,10), NO cell mask       VERIFIED     same; our w_ mask IGNORED
    optimiser         AdamW, OneCycle, bb 3e-5 / head 1e-3, wd 0.02          VERIFIED     defaults
    schedule          16 ep, bs 8, k 12, k_eval 24, grad_ckpt                PARTIAL      the header's "typical
                                                                                          invocation"; the exact
                                                                                          scored command is absent
    normalisation     imagenet at inference                                  VERIFIED     --norm imagenet (train =
                                                                                          infer; script default is
                                                                                          "none" for raptor SSL)
    augmentation      intensity x U(0.9,1.1), NO flips                       VERIFIED     same
    selection         gold58 macro-AUC every epoch, best epoch kept          VERIFIED     same; the scored file is
                                                                                          raptor_ft_<tag>.pt = the
                                                                                          BEST-EPOCH checkpoint, not
                                                                                          the top-3 average
    data              all 4,349 non-gold studies, no CV                      VERIFIED     same
    inference         K_EVAL=42, fp16 autocast, rankpct, one arm             VERIFIED     their inference code
**Two components are UNAVAILABLE and both are ones the author calls central** (the docstring opens with
the labels: "worth saying up front, because it shapes everything else"). So this is a PARTIAL
RECONSTRUCTION. It is NOT expected to reproduce 0.924, and a shortfall does not mean the recipe is weak.
Rule compliance: their CODE is used; every WEIGHT is trained by us from a public ImageNet checkpoint;
labels_llm_public is the public CC0 label set (per the user, acceptable as training data).

## What the author's own notes tell us to expect -- recorded because they bound the result
- "on the live leaderboard CoAtNet alone scored 0.914 while every blend scored 0.914-0.915, so ensembling
  is worth ~+0.001 there" -- independent confirmation of our ensemble-saturation finding.
- The 0.914 -> 0.924 step came from DATA: corpus 3,200 -> 4,349 labelled studies (+37.8%), gold58
  0.8923 -> 0.9054. So training-set size is a ~0.01-scale lever for them. Our CV members train on ~3,525.

## Arm
    R = their training script, UNMODIFIED, run with:
        --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --epochs 16 --bs 8 --k 12 --k_eval 24
        --grad_ckpt --norm imagenet --folds 0 --tag rsnaR --labels <labels_llm_public, dense>
    on a corpus built by THEIR build_study over train_series.csv (4,407 studies; gold58 held out by the
    script itself). Deployed file: raptor_ft_rsnaR.pt (best epoch), through THEIR inference code.

## Stage 1 -- the question asked: does their recipe, trained by us, reach their range? (ONE LB read)
Submit R ALONE. Outcomes, fixed now:
    R >= 0.914   reproduces their range (their pre-expansion LB was 0.914) despite the two missing
                 components -> fund Stage 2.
    0.905 <= R < 0.914   weaker than theirs, but within 0.005 of H5 (0.910): comparable strength and a
                 wholly different pipeline -> still a candidate partner -> fund Stage 2.
    R < 0.905    too weak to help a blend by our own rule (members must be within ~0.005 standalone).
                 Record that the missing SSL init and/or labels are the suspects; STOP.
ALSO RECORD from that submission: its **scored runtime**. R needs a SECOND decode path (64 slices,
140 mm, 6-94%) that H5 does not share, so R's runtime decides whether Stage 2 fits in 9 h at all.
Not a gate, and not trusted: R's gold58 score. labels_llm_public was tuned on gold58 by its publisher
AND gold58 selects the epoch, so it is doubly optimistic.

## Stage 2 -- only if Stage 1 funds it
H5 + R, judged on the LB. R has no OOF (full data), so there is NO offline composition gate; that is the
cost of testing the recipe whole. To keep this from becoming LB tuning: the composition weight will be
registered BEFORE the H5+R submission, using only R's standalone LB, and **exactly one** H5+R
composition will be submitted. No weight search on the leaderboard.

## Standing limits
H4 (0.907) and H5 (0.910) remain the selected submissions and are untouched. Oct 14 hard abort,
Oct 15 merger deadline. Cost estimate: corpus ~32 GiB / <1 h CPU; training ~1.5-2 GPU-h on the H100;
one Kaggle submission.

## TRAINING RESULT (2026-09-22 13:11 JST) -- offline diagnostics only; the registered gate is the LB
16 epochs, 8283 s (2.3 h). train 4349 | gold-val 58 (the author's exact counts). Best epoch 11.
    gold58 macro-AUC   R 0.8980 (best epoch) | top-3 SWA 0.8985 | epoch-ensemble 0.8999
    curve  ep8 0.894 ep9 0.894 ep10 0.896 ep11 0.898 ep12 0.892 ep13 0.893 ep14 0.894 ep15 0.893
On the author's own 58-study gate: their 3,155-study model 0.8923 (-> LB 0.914), their scored 4,349-study
model 0.9054 (-> LB 0.924). R lands BETWEEN them. gold58 cannot resolve under ~0.04, and R's number is
optimistic (the epoch was picked on these 58; the plateau ep8-15 averages ~0.894), so this says only
"not obviously broken, plausibly in their range".
Same-gate context (exploratory): H5 5-fold OOF on gold58 = 0.8874; rho(R, H5) = **0.838**, lower than
any pair inside H5 (0.865-0.909). Per-label R beats H5 on Lateral Meniscus +0.051, Baker's +0.044,
Medial OA +0.033, Contusion +0.032, Fracture +0.027; H5 beats R on MCL +0.079 (9 positives). The
pattern matches the author's attribution of Lateral Meniscus/Fracture gains to the wide 6-94% span.
A 50/50 rank blend scores 0.9066 on gold58 -- NOT a Stage-2 estimate: 58 studies, and R's epoch was
selected on those same studies, so the blend inherits that selection.
Submission dry-run: the exact kaggle_kernel_dreadR script's functions loaded our checkpoint through
THEIR load_model and scored the 3 local test studies (42 windows each, finite, 12/12 labels vary).
H100: ~1 s decode + ~0.15 s inference per study. T4 inference is expected several times slower.
Deployed per registration: raptor_ft_rsnaR.pt (best epoch), NOT the SWA file -- that matches the scored
reference's file type, and SWA's +0.0005 on gold58 is noise.

## STAGE 1 RESULT (2026-09-22) -- R standalone **0.928 public LB**, runtime ~1 h. -> FUND Stage 2.
Submitted as "Submission_expert_H4 - Version 17". Registered rule: R >= 0.914 reproduces their range
-> fund Stage 2. R cleared it by 0.014 and also exceeded the reference it reconstructs (0.924).
Pre-submission parity: Kaggle's commit ranked the 3 public test studies IDENTICALLY to our local run on
12/12 columns, despite different pydicom/OpenCV/torch versions (torch 2.10 on Kaggle).
READING IT:
- **+0.018 over H5 (0.910) from ONE model** -- 12x the gain H5 bought with a fifth member.
- **The two UNAVAILABLE components were not necessary.** Without the author's raptor SSL init and without
  the author's labels (public CC0 soft labels instead, dense), the recipe reached 0.928. 0.928 vs 0.924
  is within public-LB noise; do NOT claim we beat the reference, only that we reproduced its range.
- **Our offline process could not have found this.** Every ingredient tested alone on our T1 was null or
  negative (cnmxmil -0.0115, MaxSpan-on-DINOv2 -0.031, public labels -0.0008, MIL tie). The bundle, judged
  on the LB, is +0.018. Either the ingredients interact (the bundle exceeds the sum of its parts), or T1
  misjudged them one by one, or both -- this experiment cannot separate those. It does establish that
  "closed on T1, one ingredient at a time" was NOT evidence the recipe could not work.
- gold58 pointed the right way (R 0.8980 vs H5 0.8874, +0.011) -- ONE data point, R was also selected
  on gold58, so this does not rehabilitate gold58 as a selector.
- Runtime ~1 h standalone. H5 alone was projected 5.5-5.8 h, so H5 + R is ~6.5-7 h against 9 h.
