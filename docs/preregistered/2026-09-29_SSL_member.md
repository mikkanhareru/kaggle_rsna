# SSLM: does the OLD pipeline's MAE member still add to the current RGR blend? -- offline, zero GPU

Registered 2026-09-29 15:14 JST, BEFORE any blend number is computed. User's question; the inputs already exist.

## What is being reused
M = the MAE-pretrained DINOv2 ViT-S/14 member of H5 (rsna-lb-findings: "failed as a SWAP, passed as a 5th MEMBER",
five-fold +0.00152, H4 0.907 -> H5 0.910 public, 2026-09-22). Its five-fold OOF predictions are on disk at
artifacts/runs/f{k}_vit_small_patch14_dinov2_lvd142m__slotnbdp__s4mae_fine/oof.parquet.
CHECKED before registering: those files cover every study in the CURRENT held-out populations (fold 0 869/869,
fold 4 870/870; the SSL files hold 881/882, i.e. a superset), so the two pipelines' fold maps are compatible and the
contrast can be computed on exactly the current held-out sets. Predictions are restricted to those ids, nothing else.

## The prior, stated before looking
M belongs to a pipeline whose best public score was 0.910; the current RGR blend is 0.934. That is a ~0.024 gap,
and our own measured relation between a member's strength gap and its equal-weight contribution is
    REG 0.004 behind -> +0.0006 | R16 0.007 -> -0.0002 | VMamba-T 0.015 -> -0.0021
so at EQUAL weight a member ~0.024 behind is expected to lose clearly. That is why equal weight is not the primary
here: H5 gave M a 20% weight, and 20% is the setting that was actually validated for this member.

## Decision rule, frozen
PRIMARY: B20 = 0.8*rank(CTRL) + 0.2*rank(M) vs CTRL, where CTRL = rnk_RNK_fold{f}.npz; 12-finding macro fused T1,
held-out, paired study bootstrap (4,000 draws/fold), folds 0 and 4.
    PASS = mean >= +0.001 AND both folds positive -> compute all five folds; confirmation as elsewhere
    (five-fold mean >= +0.001, folds 1-3 mean > 0, CI excludes 0), then gold58 as a veto.
REPORTED, never gates: B at 0.5/0.5 and 0.9/0.1; M's standalone macro vs CTRL; Spearman M vs CTRL; per-label deltas.
DEPLOYMENT CONDITION, registered now because it can veto a passing result: M does NOT share the current 64-slice
corpus. It consumes the old pipeline's slot/native-neighbour inputs, so shipping it means running a SECOND
preprocessing pipeline inside the submission notebook. Preprocessing is ~84% of per-study runtime, so this is the
expensive part, not the model. A pass therefore requires, before any submission: the second pipeline reproduced in
the notebook, its per-study cost measured on the T4, and the total still inside the 9 h limit with the 4 RGR arms.
If that cannot be shown, the offline result is recorded and nothing ships.

## RESULT (2026-09-29 15:35 JST): PRIMARY PASSES on both rulers -- the largest offline effect measured in this project
Fold-map check done BEFORE the numbers: the SSL oof files hold 881/882 studies per fold against our 869/870, and
the union of the extras over five folds is EXACTLY the 58 gold studies. For the 4,349 non-gold studies the two
pipelines' fold maps agree, so M's predictions on our held-out sets are genuinely out of fold.
    B80 - CTRL (PRIMARY): mean +0.00566, CI [+0.00483, +0.00651]; f0 +0.00567 f1 +0.00533 f2 +0.00607 f3 +0.00543
                          f4 +0.00581; folds 1-3 +0.00561 -> five-fold rule met on every clause.
    B50 - CTRL (reported): +0.00839, CI [+0.00649, +0.01036]   B90: +0.00318   M standalone: -0.00207 (CI spans 0)
    Spearman M vs CTRL 0.816-0.860 -- far below VMamba's 0.916 and same-recipe retrains' 0.94-0.96.
SSL EXPOSURE, the one real threat: MAE pretraining used folds 0-3 and reserved fold 4. Fold 4 is therefore the only
clean fold, and it shows +0.00581 (B80) / +0.00900 (B50) -- at or above the contaminated folds. Image exposure is
not what produces the gain.
gold58 VETO CHECK (host labels, n=58, each gold study scored by the fold model that held it out, on BOTH sides):
    RGR 0.9060 | SSL 0.8733 | B80 0.9091 (+0.0031) | B90 0.9072 (+0.0012) | B50 0.9052 (-0.0008)
The registered primary B80 passes. B50, whose T1 gain is the largest, reads flat on gold58; with n = 58 that cannot
resolve +-0.004 and is not evidence against it, but it is not evidence for it either. The registered weight is
kept: after the metadata episode (T1 +0.0021, gold58 -0.010, LB -0.015) a larger T1 number is not a reason to
change a weight after the fact.
MY PRIOR WAS WRONG, recorded: this file predicted M would be ~0.024 behind CTRL because its pipeline's best public
score was 0.910 against RGR's 0.934. Measured, M is 0.002 behind. An ENSEMBLE's leaderboard score says nothing
about the standalone strength of its members; the member must be scored on the current ruler directly.
STATUS: offline confirmed; NOT shipped. The registered deployment condition is unmet until the second preprocessing
pipeline is reproduced in the notebook and its T4 cost measured inside the 9 h limit.

## SHIPPING CONFIGURATION (2026-09-29 15:46 JST, written BEFORE the notebook exists or is scored)
Decision by the user: drop RGR seeds 44 and 45 from the shipping blend and spend the freed runtime on the SSL arm.
Justification, from the leaderboard: the 4-seed blend scored 0.934, exactly the 2-seed blend's score, so seeds 3 and
4 buy nothing measurable while costing ~1.5 h of the 9 h limit.
    SHIPPED BLEND = 0.4*rank(RGR s42) + 0.4*rank(RGR s43) + 0.2*rank(SSL)
This is the registered ensemble rule applied to two recipes: RGR takes 0.8 split equally over its two seeds, SSL
takes 0.2 -- the B80 weight this file registered and which passed both rulers. The weight is NOT re-tuned.
SSL arm: pareru806/kaggle-bundle-slot-s4mae, which is self-contained (rsna/data preprocessing, rsna/train model code,
fold0-4.pt, manifest with per-checkpoint fingerprints; backbone vit_small_patch14_dinov2.lvd142m at 336 px, head
"hier", 5 folds averaged as in H5). The H5 notebook (kaggle_kernel_H5, LB 0.910) already ran this preprocessing on
Kaggle, so the path is known to work inside the limit.
Runtime plan and its check: the current notebook re-runs preprocessing PER ARM. The two RGR arms will share ONE
decode, which must be verified to leave their predictions BITWISE unchanged (build_study is deterministic, so this
is a refactor with an exact test, not an approximation). The SSL arm brings its own decode. Before any submission:
local CPU end to end, then a free Kaggle commit run reporting per-stage wall time; the total must fit 9 h with
margin, and every checkpoint's sha256/fingerprint must verify.
READING vs the current best (0.934, the 2-seed and 4-seed RGR blends):
    >= 0.937   the offline +0.0057/gold58 +0.0031 transferred; this becomes the shipping base and pick A
    0.935-0.936 transferred at reduced size, consistent with our usual offline->LB shrinkage; still promoted
    0.934      UNRESOLVED at 3 decimals
    <= 0.933   did not transfer -> the offline result stands recorded, the blend is not promoted, and the
               cross-pipeline direction is re-examined before any further work on it
Correction recorded here because it affects how pilots are read: 2026-09-28_VMamba_pilot.md claimed a pilot against
a single fold model "tests the harder case" for a new member. That is backwards -- a weaker control is EASIER to
improve on, so such pilots are, if anything, optimistic. It did not change the VMamba verdict (it failed anyway),
and for this member the point is moot on the leaderboard evidence that 2-seed and 4-seed RGR score identically.

## Notebook built and verified locally (2026-09-29 15:59 JST, before any Kaggle run)
kaggle_kernel_dreadRgrSsl/rsna_dread_Rgr_ssl.py, 828 lines, assembled from two sources that have each scored on the
leaderboard, with exactly these edits and no others:
  PHASE 1 (from kaggle_submission_H5.ipynb, LB 0.910), 5 edits: GENERALIST and SPECIALIST -> "s4mae";
    OUTPUT_OWNERSHIP -> {}; MEMBERS -> []; the H5 main loop replaced by an s4mae-only loop that keeps H5's
    within-bundle rule (rank inside each fold, then average the ranks) and frees the models; and the specialist
    ownership check SKIPPED with an assertion that OUTPUT_OWNERSHIP is empty -- there is no specialist here, so
    that check's premise does not hold and it is not satisfied with a fake one.
    H5's bundle discovery, cache-spec digest assertions, per-checkpoint fingerprint checks and the all-0.5
    placeholder submission (a crash-safety net that phase 3 overwrites) are untouched.
  PHASE 2 (from rsna_dread_Rgr_blend.py, LB 0.934), 3 edits: ARMS keeps seeds 42 and 43; the output stage becomes
    the registered blend; the entry point runs main() directly since phase 1 precedes it.
Name-collision audit before assembly: of 35 RGR and 87 H5 top-level names, 10 collide and 8 are plain imports;
only CROP_MM and SLOTS are real, and phase 2 redefines both for itself after phase 1 has finished with them.
LOCAL CPU END TO END on 3 studies (a /kaggle/input mirror of the shim):
  s4mae 5 folds loaded, fingerprints OK, native_nb decode, 0 fallbacks
  both RGR arms' sha256 OK
  RGR arm probabilities BITWISE IDENTICAL to the verified 2-seed notebook run on the same studies (max |d| 0.0)
  output == 0.4*rank(RGR42) + 0.4*rank(RGR43) + 0.2*rank(SSL) exactly
  the SSL arm moves 15 of 36 rank cells vs the RGR-only blend, i.e. it is wired up
Next and last check before submitting: a free Kaggle commit run for the per-stage wall time, which must leave the
scoring run inside 9 h.

## LEADERBOARD RESULT (2026-09-29 21:54 JST): 0.935 -> band "0.935-0.936", PROMOTED
RGR s42 + s43 + SSL s4mae at 0.4/0.4/0.2 scored 0.935 public, against 0.934 for both the 2-seed and 4-seed RGR
blends. Applied as registered: transferred at reduced size, still promoted -> this is the new shipping base and the
current pick A. Submission ref 56666987; the scoring run finished well inside the 9 h limit.
TRANSFER RATE, recorded because it is the smallest we have seen: offline +0.00566 (five-fold T1) and gold58 +0.0031
became +0.001 on the public split, i.e. roughly 1/5. Our earlier promotions transferred at ~1:1 to 1/2
(native-neighbour +0.0017 -> +0.001; cnxd3lo +0.00105 -> +0.001; H5 +0.00152 -> +0.003). Three readings that this
1/5 is NOT evidence the offline measurement was wrong: (a) 0.935 vs 0.934 is one step of 3-decimal display, so the
true difference lies anywhere in ~[+0.0005, +0.0015]; (b) the public split is ~325-650 studies, whose sampling SD we
estimated at ~0.003 -- larger than the effect being read; (c) the offline control was a single RGR fold model while
the shipped control is a 2-seed blend, which is stronger, so the deployed headroom was always smaller than the
pilot's. No weight is re-tuned on the basis of this reading.
