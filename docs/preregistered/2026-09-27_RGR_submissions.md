# RGR full-data submissions of 2026-09-27 -- readings written BEFORE any of them is pushed or scored

Written 2026-09-27 09:41 JST. Public references: R_swa 0.929 (incumbent single), REG 0.926, R_swa + REG blend 0.931 (LB-best).
Weights (own, full data, REG + rank lambda 0.5 tau 1.0): RNKF = raptor_ft_rsnaRgrFull_swa.pt seed 42, sha256
309b47f7...; RGR43 = raptor_ft_rsnaRgrFullS43_swa.pt seed 43, sha256 c337a569.... Datasets (private):
pareru806/kaggle-bundle-dread-rgr, pareru806/kaggle-bundle-dread-rgr43. Notebooks = the REG notebook with ONE changed
ARMS block each: rsna-knee-dread-rgr-fast (RNKF), rsna-knee-dread-rgr43-fast (RGR43), rsna-knee-dread-rgr-blend
(both, equal-weight rank-mean fixed in advance). Local CPU end to end, 3 public test studies: every sha256 OK; each
output == the weighted rank-mean of its arms; the blend's arm predictions are bitwise equal to the single notebooks'.

1. RNKF single -- THE registered leaderboard comparison of 2026-09-25_R_rank.md (addendum reading, unchanged):
     >= 0.930 consistent with the offline evidence | 0.927-0.929 UNRESOLVED | <= 0.926 red flag for the REG line
2. RGR43 single (optional) -- gapS = LB(RGR43) - LB(RNKF) is reported as ONE observed seed difference; no reading,
   no noise band (2026-09-26_full_seed_spread.md correction 4).
3. RGR blend (RNKF + RGR43, weights fixed) -- vs the LB-best 0.931:
     >= 0.932        new LB-best; candidate for the final-pick hedge (B)
     0.930 - 0.931   UNRESOLVED against the R_swa + REG blend
     <= 0.929        a two-seed RGR blend reads no better than the R_swa single: red flag, read together with (1);
                     offline-first still holds, but no REG-line model is promoted on LB grounds and the deferred
                     R43 / REG43 paired check moves back up the queue
   Reported with it, not interpreted as a seed-ensemble gain estimate (one observation): blend - max(singles) and
   blend - mean(singles) when (1) and (2) exist.
Weights of (3) are never changed according to which seed scores higher.

## Pushed and verified (2026-09-27 ~09:50 JST, user-approved)
Datasets created (private, ready): kaggle-bundle-dread-rgr (raptor_ft_rsnaRgrFull_swa.pt 292,829,899 B),
kaggle-bundle-dread-rgr43 (raptor_ft_rsnaRgrFullS43_swa.pt 292,831,375 B). Notebooks v1 pushed; the blend push was
first refused ("maximum batch GPU session count of 2") and pushed after the first commit run finished. Kaggle commit
runs (3 public studies): rgr-fast "sha256 OK 309b47f7b288"; rgr43-fast "sha256 OK c337a569be03"; rgr-blend both,
w 0.5/0.5; no FALLBACK; every submission.csv equals the local run cell for cell (max|d| 0.0).
Submissions are made manually by the user from the web UI.

## LEADERBOARD RESULTS (2026-09-27 ~12:00 JST; submitted versions pulled back: code identical to local, datasets,
## T4 and internet-off as pushed)
1. RNKF single 0.929 -> 0.927-0.929 band: UNRESOLVED. The offline evidence (+0.00431 over REG) stands; no contradiction
   claimed, no transfer shown by this reading.
2. RGR43 single 0.932 -> gapS = +0.003 (ONE observed seed difference, same recipe, same data). Reported only; it is
   the size the earlier noise estimate assumed, which neither calibrates nor confirms that estimate.
3. RGR blend 0.934 -> ">= 0.932": NEW LB-BEST; candidate for the final-pick hedge (B).
   blend - max(singles) = +0.002; blend - mean(singles) = +0.0035 (one observation of a two-seed blend on ~325-650
   public studies; not an estimate of the seed-ensemble gain).
Context: R_swa 0.929, REG 0.926, R_swa+REG blend 0.931. Both two-member blends so far read above both of their
members on the public split. No weights are changed in response to these readings.
