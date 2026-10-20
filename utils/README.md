# utils — how the efficiency changes were proven safe and measured

## verification/
| File | What it proves |
|---|---|
| `prefetch_verify.py` | The original 1-thread prefetch (2026-09-23): byte-identical inputs, identical predictions, 1.31x. |
| `poolverify.py` | Process pool (+ optional prefetch, functions exec'd from the kernel file itself): ABBA timing on 4 pinned CPUs, byte-identical vol/mask on 99 stratified studies, prediction diff within GPU repeatability, and three injected failures (worker exception, worker death, worker hang) isolated. `--ker`, `--W`, `--pf`. |
| `v3test.sh` | v3 on a 60-study fake test root: natural / two replicas / replica 1 failing / replica 0 failing must all reproduce v1's submission exactly. |
| `parity_tp.py` | A kernel's model code reproduces the trainer's saved gold58 predictions; H100 forward ms/study per backbone x K. |
| `forktest.py`, `forktest2.py` | Minimal reproduction of the cv2 fork deadlock, and proof the parent-side setting is safe and inherited. |

## measurement/
| File | What it measures |
|---|---|
| `kcurve.py` | Paired K_EVAL curve on RGR five-fold OOF (+ RN50D folds 0/4): same checkpoint, same studies, only K changes. |
| `kaggle_probe_v1/` | Kaggle-side timing on ~64 cold TRAIN studies: serial vs pooled prep, T4 forward per backbone x K. Writes no submission. |
| `kaggle_probe_v2/` | Prep-only: pool width x prefetch variants on disjoint cold sets. DO NOT SUBMIT. |
| `subpoll_jst.sh` | Read-only submission-status poller (JST timestamps, 2-min resolution) — the source of every runtime we report. |

Efficiency-LB rank bound used in the docs: the LB shows rank and AUC but not runtime; since E >= -AUC/G and the
table is sorted by E, the first listed rank whose AUC is below `ours - G * minutes / 540` bounds our rank from above.
