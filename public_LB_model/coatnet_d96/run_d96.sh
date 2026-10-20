#!/usr/bin/env bash
# run_d96.sh <seed> -- DENSE96 (96 slices, native 384) on a FULL-DATA CoAtNet-2 @384 run.
#   Registered: competition_details/preregistered/2026-10-08_DENSE96_native384.md
# The command is the logged argv of rsnaRgrFull (logs/full_RNKF.log, script_sha256 19122d15...)
# with --tag changed and the corpus pointed at the d96 files by two env vars. Labels, LR, wd,
# epochs, k, k_eval, drop_path and the rank loss are ALL unchanged from the LB-0.929 control.
# The command is a bash ARRAY and is never eval'd (handover rule 8).
set -u -o pipefail
SEED="${1:?usage: run_d96.sh <seed>}"
case "$SEED" in 42|43) ;; *) echo "!! seeds 42 and 43 only" >&2; exit 2 ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/gp.sc.cc.tohoku.ac.jp/wanrencheng/anaconda3/envs/rsna/bin/python
T="$ROOT/dread/train"
TAG="rsnaD96F${SEED}"
LOG="$ROOT/logs/d96F${SEED}_train.log"
TRAINER=knee_train_d96.py
TRAINER_SHA=98be0791abdabec19610b3e9a1817a8d0f58ce008d23532fdde678bb8dfbb30b
export RSNA_CORPUS="$T/all_vols_d96.npy"
export RSNA_MASKS="$T/all_masks_d96.npy"

die() { echo "!! $*" >&2; exit 1; }

got=$(sha256sum "$T/$TRAINER" | cut -d' ' -f1)
[ "$got" = "$TRAINER_SHA" ] || die "$TRAINER sha256 $got != registered $TRAINER_SHA"
for f in all_vols_d96.npy all_masks_d96.npy all_ids_d96.npy; do
  [ -f "$T/$f" ] || die "d96 corpus missing $f -- the build must have been interrupted"
done
[ -f "$ROOT/dread/labels_public_dense.parquet" ] || die "labels missing"
[ -f "$ROOT/artifacts/labels/labels_llm_public.parquet" ] || die "rank mask missing"

# ---- the d96 corpus must be COMPLETE and must line up with the production study order ----
"$PY" - "$T" <<'PYEOF' || die "corpus integrity check failed"
import sys, numpy as np
from pathlib import Path
T = Path(sys.argv[1])
V = np.load(T/"all_vols_d96.npy", mmap_mode="r"); M = np.load(T/"all_masks_d96.npy")
I = np.load(T/"all_ids_d96.npy", allow_pickle=True)
Io = np.load(T/"all_ids.npy", allow_pickle=True)
assert V.shape[1:] == (96, 384, 384), f"d96 vols shape {V.shape}"
assert V.shape[:2] == M.shape[:2], f"vols {V.shape} vs masks {M.shape}"
assert np.array_equal(np.asarray(I, dtype=str), np.asarray(Io, dtype=str)), "study order differs"
nv = M.sum(1); empt = int((nv == 0).sum())
assert empt == 0, f"{empt} EMPTY studies -- the build did not finish"
print(f"[d96] corpus OK {V.shape} | filled slices/study mean {nv.mean():.1f} "
      f"min {nv.min()} max {nv.max()} of 96 | 0 empty | ids match production", flush=True)
PYEOF

for f in "$T/raptor_ft_${TAG}.pt" "$T/raptor_ft_${TAG}_swa.pt" "$T/raptor_ft_${TAG}.json" "$LOG"; do
  [ -e "$f" ] && die "$f exists -- outputs are never overwritten"
done

"$PY" - <<'PYEOF' || die "GPU real-allocation test failed"
import torch
x = torch.randn(4096, 4096, device='cuda', dtype=torch.float16); y = x @ x
torch.cuda.synchronize()
free, total = torch.cuda.mem_get_info()
print(f"[d96] GPU OK {torch.cuda.get_device_name(0)} | free {free/2**30:.1f} GiB / {total/2**30:.1f} GiB", flush=True)
assert free > 40 * 2**30, f"only {free/2**30:.1f} GiB free"
PYEOF

CMD=("$TRAINER"
     --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k
     --res 384 --epochs 16 --bs 8 --k 12 --k_eval 24 --grad_ckpt --norm imagenet
     --folds 0
     --tag "$TAG" --seed "$SEED"
     --labels "$ROOT/dread/labels_public_dense.parquet" --workers 16
     --drop_path 0.1
     --rank_w 0.5 --rank_tau 1.0
     --rank_mask "$ROOT/artifacts/labels/labels_llm_public.parquet")

cd "$T" || die "cannot cd $T"
{
  echo "=== DENSE96 full-data seed ${SEED} (2026-10-08_DENSE96_native384.md) ==="
  echo "started      $(TZ=Asia/Tokyo date '+%Y-%m-%d %H:%M:%S JST')"
  echo "trainer sha  $got  (base 19122d15..., ONE substitution: corpus+mask env override)"
  echo "RSNA_CORPUS  $RSNA_CORPUS"
  echo "RSNA_MASKS   $RSNA_MASKS"
  echo "control      rsnaRgrFull on the 64-slice 336 corpus, LB 0.929 (R_swa)"
  echo "note         res 384 on a NATIVE 384 corpus -> F.interpolate is skipped entirely"
  echo "command      $PY ${CMD[*]}"
} > "$LOG"

"$PY" "${CMD[@]}" >> "$LOG" 2>&1 &
PID=$!
echo "launcher pid  $$"   >> "$LOG"
echo "train pid     $PID" >> "$LOG"
[ -r "/proc/$PID/cmdline" ] && tr '\0' ' ' < "/proc/$PID/cmdline" > "$ROOT/logs/d96F${SEED}_launch.txt"
echo "[d96] seed ${SEED} launched train pid $PID -> $LOG"
wait $PID
rc=$?
echo "=== DENSE96 ${SEED} done rc=$rc $(TZ=Asia/Tokyo date '+%Y-%m-%d %H:%M:%S JST') ===" >> "$LOG"
echo "[d96] seed ${SEED} finished rc=$rc"
exit $rc
