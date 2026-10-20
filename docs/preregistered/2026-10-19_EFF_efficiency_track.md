# EFF: an Efficiency-Prize candidate, chosen by a rule frozen BEFORE the K-curve and pool numbers exist

Registered 2026-10-19 ~14:50 JST, before `logs/kcurve.csv` and the process-pool verification had printed
anything. Authorised ("轻量化一下模型，争取拿一个efficiency LB").

## The metric, and what it makes cheap
    E = AUC / (Benchmark - maxAUC) + RuntimeSeconds / 32400        (minimise)
With G = maxAUC - Benchmark ~ 0.45 (ASSUMED: benchmark = constant sample submission 0.5, private max ~0.95),
**one minute of runtime costs as much as 0.00083 AUC**; one hour = 0.050 AUC. Eligibility: the submission must
be one of the two final selections. Existing scored candidates under G = 0.45:
    d96 single     0.935 @ ~120 min   E = -1.856
    RGR43 single   0.932 @  ~60 min   E = -1.960      <- the current best efficiency pick, with zero new work
    R_swa_fast     0.929 @  ~60 min   E = -1.953

## Measured before this registration (2026-10-19)
* Kaggle scoring is **GPU-bound and linear in windows**: R (K42) ~60 min and d96 (K86) ~120 min give 85.7 vs
  83.7 s per window position over the hidden test, and the H100 forward ratio K43/K86 = 0.51 predicts R at ~61 min.
* RN50D forward is **3.20x / 3.44x / 3.66x** cheaper than CoAtNet-2 at K 86 / 43 / 24 (H100, fp16, kernel code).
* RND96 kernel code reproduces the trainer's gold58 predictions (min column Spearman 0.998; CoAtNet control 0.9996).
* Below the GPU time sits a **preparation floor** (DICOM header+pixel decode, crop, resize), single-threaded in
  every kernel we have shipped. Its Kaggle size is NOT measured; only a process pool can lower it.

## Candidate grid and the frozen selection rule
Grid: arch in {CoAtNet-2 d96 s42 (LB 0.935 at K86), RN50D d96 s42} x K_EVAL in {86, 62, 43, 36, 24, 18}.
    GPU(arch,K)  = 120 min x t_H100(arch,K) / t_H100(CoAtNet,86), t interpolated linearly in K between measured points
    dAUC(K)      = RGR five-fold paired OOF delta vs full coverage (kcurve), mapped to DENSE96 by COVERAGE:
                   d96 K  <->  RGR K x 62/92 (valid window positions per volume), linear interpolation
    AUC(CoAtNet,K) = 0.935 + dAUC(K)
    AUC(RN50D,K)   = 0.935 - g + dAUC_RN(K), g = paired OOF gap RN50D-vs-RGR at K42 on folds 0/4 (the 2026-10-02
                   RN50D checkpoints, res 320: a pessimistic proxy for the res-384 d96 model -- stated, not fixed)
    PREP         = 25 min x (pool ms/study / thread ms/study) / 0.40   [i.e. 25 min if the pool gives 2.5x]
                   (the 25 min is an ASSUMPTION for a 3-process pool on Kaggle; it is the number this run can falsify)
    wall         = max(GPU, PREP) + 2 min (start-up, model load)
    E_pred       = -AUC / 0.45 + wall / 540
**Pick argmin E_pred.** Ties within 0.005 E (~2.7 min) go to CoAtNet (LB-verified family), then to larger K.
The process pool ships ONLY if `poolverify.py` prints ADOPT (inputs bitwise-identical on 99 stratified studies,
predictions within GPU repeatability, both injected failures isolated, faster than the deployed thread).

## Judgement (one submission; no chase)
Observed: public LB AUC and the submit-to-score elapsed time (the only runtime Kaggle shows us; includes queue).
* The EFF kernel replaces RGR43 as the efficiency pick iff its E (G = 0.45, elapsed time) is lower than -1.960.
* If elapsed time exceeds 1.5x the predicted wall, the PREP floor was underestimated; this is RECORDED, and at
  most ONE pre-declared follow-up is allowed: the next grid point with the same arch and smaller K.
* No weight or K tuning against the public LB beyond that single step.

## AMENDMENT 1 (2026-10-19, still BEFORE any kcurve / poolverify output) -- the GPU-bound claim is withdrawn
The single-arm d96 kernel (LB 0.935, ~2 h) has **no prefetch at all**: its loop is build -> windows -> GPU,
serially. Only the R_fast / RGR lineage got the verified 1-thread prefetch. So its ~2 h is prep + GPU SUMMED,
and the "85.7 vs 83.7 s per window" agreement above is not evidence of GPU-bound scoring: with a DENSE96 prep
cost ~1.26x R's (more slices decoded, larger crops), the same two rough runtimes fit "mostly GPU" and "half
prep" equally well. The H100 profile (prep 0.45 s vs GPU 0.084 s at K42) plus a T4 that is ~6-10x slower
than an H100 points to prep and GPU being COMPARABLE on Kaggle.
Consequences for the rule (formula unchanged, inputs replaced):
* GPU(arch,K) and PREP are taken from a **Kaggle timing probe** (a private kernel run on ~60 TRAIN studies,
  no submission slot): serial per-study prep on cold reads, T4 forward per arch x K, and pool prep. The old
  "120 min x H100 ratio" and "25 min" inputs are used ONLY if the probe is not run.
* A zero-AUC-risk floor is now explicit: the scored d96 model + the verified 1-thread prefetch (or the pool)
  changes WHEN preparation runs, not what it computes.

## RESULT 1 (2026-10-19 ~15:40 JST): rule inputs measured, rule applied, candidate selected
**Kaggle timing probe** (`rsna-knee-eff-probe` v1; Tesla T4, 4 vCPU Xeon @2.0GHz; 30 cold TRAIN studies, medians, ms):
    serial prep 2358 (mean 2838) | pool W=3 prep 907 (3.13x)
    CoAtNet-2  K86 1445  K43 707  K24 403      RN50D  K86 386  K43 172  K24 98      windows K86 191 K43 104 K24 59
Calibration: the scored serial d96 kernel costs 2838+191+1445 = 4.47 s/study, so ~2 h <-> N ~ 1,610 hidden studies,
and **63% of that kernel's runtime was DICOM preparation**, not the model.
**K-curve** (`logs/kcurve.csv`, RGR five folds, paired, public ruler; vs K62 = full coverage):
    K42 -0.00125 | K24 -0.00341 | K12 -0.01933   (5/5 folds monotone)
    RN50D (folds 0/4, res 320) vs RGR at K42: g = 0.0187
**Rule applied** (wall = N x max(907, windows+GPU) + 2 min; dAUC mapped by coverage 62/92):
    CoAtNet K86 0.9348  45.9 min  E -1.992 | K62 0.9337 33.5 min E -2.013 | **K43 0.9322 26.3 min E -2.023**
    CoAtNet K36 0.9316  26.3 min  E -2.022 (tie <0.005 -> larger K) | K24 0.9212 E -1.999 | RN50D K86 0.916 E -1.988
**Selected: d96 CoAtNet s42, K_EVAL 43, process pool W=3** -> `kaggle_kernel_dreadEff` (`rsna-knee-eff-d96-k43`).
Below K~46 the main thread (windows+GPU 811 ms at K43) is under the 907 ms preparation floor, so fewer windows or
a smaller backbone buy NO wall-clock and only cost AUC. The ranking K43 > K62 holds for any N > ~420.
Prediction to check on the leaderboard: public AUC ~0.932 (+-0.003), elapsed ~26 min (registered 1.5x = 39 min).

## AMENDMENT 2 (2026-10-19 ~16:20 JST, BEFORE any I/O-v2 measurement) -- faster preparation, same rule
Observed: the DICOMs are UNCOMPRESSED (Explicit VR Little Endian, ~526 KB = raw pixels), ~50 MB read per study;
serial 18 MB/s and pool 55 MB/s are far below disk bandwidth -> per-file LATENCY bound, so concurrency is the lever.
Two preparation variants, both byte-identical by construction (they change only WHEN bytes are read):
  (a) more pool workers than cores; (b) each worker first reads the selected series' files into the page cache
  with 8 threads, then runs the UNCHANGED build_study.
Probe v2 (DO NOT SUBMIT, prep only) measures W3, W6, W8, W3+pf8, W4+pf8, W6+pf8 and W3 again (drift) on
disjoint cold sets of 40 studies. Decision, fixed now:
* PREP_v2 = the best variant's ms/study, adopted only if it beats W3 by >= 15% AND the repeated W3 is within
  15% of the first W3 (otherwise drift swamps the comparison -> keep v1).
* Re-apply the SAME rule (grid, dAUC, GPU, N = 1,610, tie-break) with PREP_v2. If it selects a configuration
  different from v1 (CoAtNet K43, W3), that configuration is EFF v2 -- one submission. If it selects K43 again,
  v2 ships only if its predicted wall is >= 2 min shorter than v1's.
* The variant must pass the same local parity standard (byte-identical inputs) before shipping.

## RESULT 2 (2026-10-19 ~16:15 JST): process pool -- a fork deadlock found locally, fixed, then ADOPT
First verification run HUNG (workers 0% CPU, futex wait). Minimal reproduction (forktest*.py, 30 s timeouts):
it hangs ONLY when the parent has already run cv2's multi-threaded resize AND the forked child then calls
cv2.setNumThreads(1) in its initializer (cv2's copied thread-pool state waits on threads fork did not copy).
OpenMP-before-fork, CUDA-before-fork and torch.set_num_threads in the child were all fine.
The EFF ordering (fork at first submit, before any cv2 use) had dodged it -- the Kaggle probe and the local
end-to-end run passed -- but only by ordering. Fix, now in the kernel:
  * cv2.setNumThreads(1) in the PARENT before fork and before any cv2 call; no worker initializer (children
    inherit 1 thread -- verified cv2.getNumThreads() == 1 in the child);
  * watchdog: result(timeout=300 s); on timeout the pool is terminated and the rest is built in-process.
poolverify2 (99 stratified studies, 4 pinned CPUs): inputs byte-identical 99/99, predictions max|d| 0 with 0/1188
rank changes; injected exception -> only the victim is 0.5; injected worker DEATH -> rest built in-process,
all rows equal the clean run; injected worker HANG -> watchdog (20 s in the harness) -> all rows equal the clean
run, process exits cleanly. **VERDICT: ADOPT.** Rebuilt kernel's local end-to-end output is byte-identical to the
pre-fix build. (Local timing 1.11x vs the 1-thread prefetch is not the Kaggle number: warm cache, H100.)

## RESULT 3 (2026-10-19 16:44 JST): probe v2 -- the frozen rule says KEEP v1
`rsna-knee-eff-probe2-do-not-submit` v1 (T4 box, 4 vCPU Xeon @2.0GHz; disjoint cold sets of 40 TRAIN studies):
    W3 1728 ms (66 MB/s) | W6 964 | W8 774 | W3pf8 879 | W4pf8 577 | W6pf8 541 (170 MB/s) | W3hd8 733 | W6hd8 588 (179 MB/s)
    W3 repeated 1086 ms (74 MB/s)      byte-identity of prefetch output: 4/4 + 4/4
**Rule outcome: the repeated W3 drifted -37% (> 15%) -> KEEP v1 (CoAtNet K43, W3). No EFF v2 under this rule.**
Post-hoc observation, recorded as such and NOT used to change the outcome: the two W3 sets differ in size (114 vs
80.6 MB/study); in MB/s the W3 runs differ by 12%, and W6pf8 is -50% even against the FASTER W3 run. The drift check
was written in ms/study without normalising for set size -- a flaw in the rule, not evidence against the effect.
Any EFF v2 (re-applied rule would pick ~K36 + W6pf8, ~20 min vs 26, dE ~ 0.01) therefore needs an explicit user
override, decided AFTER EFF v1's measured elapsed time has tested the runtime model.

## RESULT 4 (2026-10-19 ~17:15 JST): EFF v2 PREPARED AND VERIFIED, NOT PUSHED (awaits an explicit user override)
`kaggle_kernel_dreadEff2` (`rsna-knee-eff-d96-k36-w6pf8`): d96 CoAtNet s42, K_EVAL 36, W=6, PF_THREADS 8; diff vs v1 =
K_EVAL, W, and the prefetch helpers only. poolverify (functions exec'd from the v2 kernel file itself, W6 + prefetch,
99 stratified studies): inputs byte-identical 99/99, predictions max|d| 0, 0/1188 rank changes, exception / worker
death / worker hang all isolated -> ADOPT. Local end-to-end run OK. EFF v1 (K43) submitted 17:02 JST; its measured
elapsed time decides whether the runtime model holds before any override is considered.

## RESULT 5 (2026-10-19 17:19 JST): EFF v1 scored
`rsna-knee-eff-d96-k43` Version 2: **public LB 0.933** (predicted 0.932 +- 0.003), submitted 17:01:55 JST (Kaggle clock),
COMPLETE seen by the poller at 17:19 JST (server clock, ~3.5 min behind Kaggle; previous poll 17:17 PENDING)
-> **elapsed ~19-21 min** incl. queue + start-up (predicted 26.3, registered ceiling 39). E(G=0.45) ~ -2.036 < -1.960:
**per the rule, EFF v1 replaces RGR43 as the efficiency pick.** K86 -> K43 cost 0.002 on the LB (K-curve said 0.0028).

## AMENDMENT 3 (2026-10-19 ~17:30 JST, BEFORE any v3 code or measurement) -- USER OVERRIDE + v3
Found in the Kaggle logs: `device cuda | gpus 2` -- every kernel so far used ONE of two T4s.
**User override (explicit, "可以，批准测试v3"):** RESULT 3's "keep v1" is overridden for the prefetch; reason on record:
the -37% W3 drift tracks set size (114 vs 80.6 MB/study; 12% apart in MB/s) and W6pf8 is -50% even against the faster
W3 run. This is the user's decision, logged as such; the rule's text is not rewritten.
**v3 = v1's predictions, faster:** d96 CoAtNet s42, **K_EVAL 43 (unchanged)**, two model replicas on cuda:0 / cuda:1
(study i -> GPU i mod 2, one thread per GPU), W6 + PF8 preparation. With twice the GPU budget the rule's choice returns
to K43 (K36 would save < 1 min of prep-bound time for -0.0006 AUC), so v3 ships v1's exact model and windows.
Safety, built in: a cross-GPU self-check at start-up (same synthetic input on both GPUs; max|d| > 1e-2 or any error ->
single GPU); a study that raises on one GPU is retried on the other; a GPU failing 3x in a row is retired; the pool
watchdog/in-process fallback from v1 is unchanged.
Predictions to check: public LB **0.933** (must be within 0.001 of v1: predictions differ only by fp16 device noise),
elapsed **~9-11 min** (from ~20). Judgement: v3 replaces v1 as the efficiency pick iff LB >= 0.932 AND elapsed is
>= 3 min shorter than v1's ~20 min. One submission; no further tuning of W / PF / K.

## RESULT 6 (2026-10-19 ~17:35 JST): v3 verified locally (60-study fake test root, 4 pinned CPUs)
`kaggle_kernel_dreadEff3` (`rsna-knee-eff-d96-k43-2gpu`). Every variant's submission vs v1's on the same 60 studies:
    T1 natural (1 GPU here)                      max|d| 0, identical, 0 fallback rows
    T2 two replicas (self-check max|d| 0)        identical
    T3 replica 1 injected failure -> retired     identical (all its studies retried on replica 0)
    T4 replica 0 injected failure -> retired     identical
cuda:1 itself can only be exercised on Kaggle: the commit run's self-check line is the first real two-GPU evidence.

## RESULT 7 (2026-10-19 18:23 JST): EFF v3 scored -- REPLACES v1 as the efficiency pick
`rsna-knee-eff-d96-k43-2gpu` Version 2 (run by the user in the editor, then submitted): **public LB 0.933** (= v1, as
predicted: same model and windows). Submitted 18:06:13 JST Kaggle clock (~18:02.7 server); poller PENDING at 18:14,
COMPLETE at 18:16 (server) -> **elapsed ~11.3-13.3 min** incl. queue + start-up (v1 ~19-21; predicted ~9-11).
Registered test: LB >= 0.932 AND >= 3 min faster than v1 -> **PASS (0.933, ~8 min faster)**. The two-GPU self-check
line was not captured (manual editor run), so the 2-GPU vs prefetch share of the ~1.6x is not separated.
E(G=0.465) ~ -1.98 vs v1 -1.97. Rank bound on the Oct-19 efficiency-LB snapshot: v1 <= 389, **v3 <= 221-230**.
FINAL-SELECTION NOTE: the efficiency pick is v3 (`rsna-knee-eff-d96-k43-2gpu` V2) and must be one of the two ticks.
