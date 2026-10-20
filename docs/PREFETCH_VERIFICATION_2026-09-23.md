# Bounded prefetch for R_swa inference -- verification record

2026-09-23. Efficiency only: model, checkpoint, 42 windows, 384 px, 140 mm preparation, fp16 and the
rank output are UNCHANGED. The scored 0.929 notebook (rsna/kaggle_submission_dreadRswa.ipynb) remains
the fallback and the accuracy reference; this is a SEPARATE artifact.

## Why: the loop was CPU-bound, not GPU-bound (rsna/train/profile_R.py, 39 studies, CUDA events)
    CPU header 0.132s (24.9%) | pixel 0.182s (34.4%) | crop 0.064s (12.0%) | windows 0.068s (12.8%)
    CPU total 0.446s (84.2%)  |  GPU forward 0.084s (15.8%)   [H100]
NOTE (external review, accepted): profile_R.py's TOTAL is assembled from selected components and misses
some preparation work -- the whole-loop measurement below (594 ms/study serial) is the adoption figure,
not the profiler's 529 ms. Overlap therefore saves MORE than the profiler predicted.

## Design
One `ThreadPoolExecutor(max_workers=1)`; a rolling submit/`result` keeps at most **1 running + 1 queued**
future. Window construction and GPU inference stay on the main thread. No `executor.map` (it can submit
the whole dataset; its `buffersize` needs a newer Python than we can assume on Kaggle).

## Verification (rsna/train/prefetch_verify.py, 99 stratified studies)
Sample: 20 lowest slice-fill, 10 four-series studies (a slot MUST be empty), 10 nine-plus-series, 59
random. **`cudnn.benchmark=True` and `allow_tf32=True` set to match the deployed script** -- the first
run of the harness omitted them (the reference sets them at module level, and the ast extraction takes
only constants and functions), so it tested under MORE deterministic conditions than deployment. Result
below is the corrected run.
    timing (ABBA, warm caches both ways)  serial 58.95 / 58.65 s  | prefetch 45.23 / 44.25 s
        -> 594 -> 452 ms/study, **1.31x, 23.9% of wall clock saved**
    peak RSS 2.10 -> 2.26 GiB (+0.16) | peak CUDA allocated 7.61 GiB
    INPUT PARITY  vol sha / mask sha / window centres identical **99/99**
    PREDICTIONS   serial vs serial   max|d| 0.0, 0/1188 rank changes   <- repeatability baseline first
                  serial vs prefetch max|d| 0.0, 0/1188 rank changes
                  prefetch vs prefetch max|d| 0.0
    FAILURE       injected whole-study failure: victim row stays the 0.5 fallback in BOTH arms; every
                  other row identical between arms AND unchanged vs the clean run; producer survived
                  and later studies scored.
cuDNN autotuning proved deterministic here in practice even with benchmark on -- established by the
serial-vs-serial baseline rather than assumed.

## Artifact
`rsna/kaggle_submission_dreadRswa_fast.ipynb` and `kaggle_kernel_dreadRswaFast/`. Diff vs the pinned
0.924 reference: 3 hunks -- the ThreadPoolExecutor import, the ARMS weights filename, and the per-study
loop. It is no longer "verbatim except one line", which is why it ships as a separate notebook.
LIMITATION: the notebook itself cannot be dry-run locally (it resolves /kaggle/input paths). The loop
logic is the same rolling submit/result pattern verified in the harness, but that correspondence is
structural, not byte-identical -- the first Kaggle commit is where it is confirmed end to end.

## Next
Re-profile AFTER prefetch before considering ordered parallel reads: with the GPU wait hidden, decode's
remaining share may look different, and decoding is not necessarily purely I/O-bound.

## KAGGLE CONFIRMATION (2026-09-23) -- the fast variant scored **0.929**, identical to the serial 0.929
End-to-end confirmation of the parity argument: the same score from a loop that only changed WHEN the
preparation work happens. The harness had shown bitwise-identical inputs and predictions on 99 stratified
studies; the leaderboard now shows the same public score on the full hidden test. The structural (not
byte-identical) correspondence between the harness pattern and the notebook loop is therefore confirmed
in the one place it could not be checked locally.
Still to record: the fast run's wall-clock, which is the actual efficiency evidence (expected ~24% below
the 0.929 run). Efficiency comparison so far: blend 0.930 in ~8 h vs R_swa_fast 0.929 in ~1 h.
