#!/bin/bash
# H5: folds 0-3 of M, the MAE-init member that passed S4 on the clean fold-4 holdout.
# Registered: competition_details/preregistered/2026-09-21_ssl_fivefold_H5.md
# Args in a bash ARRAY, never eval (an eval'd "--aug-off ''" loses the empty value).
set -euo pipefail
cd /import/fox/will/graduation_thesis/RSNA_kaggle
PY=~/anaconda3/envs/rsna/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENC=artifacts/ssl/mae_ssl1_s150000.pt
SHA=353becb85cf4053c
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$USED" -gt 5000 ]; then echo "!! GPU busy (${USED} MiB)"; exit 1; fi
[ -f "$ENC" ] || { echo "!! missing $ENC"; exit 1; }
t=$(date +%s); find cache_slots_336_130mm_12sl_nb/train -name '*.pt' -print0 | xargs -0 -P 8 -n 40 cat > /dev/null
echo "nb cache pre-read $(( $(date +%s) - t )) s"

# BYTE-IDENTICAL to the fold-4 M that passed S4; only --fold changes. No --exclude-folds.
COMMON=(--accum 1 --aug-off ''
  --backbone vit_small_patch14_dinov2.lvd142m --band-hi 0.8 --band-lo 0.2 --batch-size 16
  --cache slot --crop-mm 130.0 --ema-decay 0.999 --epochs 50
  --fine-attn True --freeze-epochs 0 --head-type hier --img-size 336 --labels '' --loss both
  --lr 0.0003 --mixup none --mixup-alpha 0.3 --n-regions 0 --n-series 6 --n-slices 4
  --native-nb True --no-augment False --overfit 0 --policy fused --rank-macro False --rank-tau 1.0
  --rank-trim 0.0 --rank-trim-warmup 5 --select-on tier1 --swa-epochs 5
  --rank-w 1.0 --target y_ --val-policy fused --warmup-frac 0.05 --weight-decay 0.05 --workers 16
  --encoder-lr-mult 0.081 --seed 0)

for F in 0 1 2 3; do
  echo "=== M fold $F start $(date -u) ==="
  $PY -m rsna.train.loop --fold "$F" --run-tag s4mae "${COMMON[@]}" \
      --encoder-init "$ENC" --encoder-init-sha "$SHA"
  D=artifacts/runs/f${F}_vit_small_patch14_dinov2_lvd142m__slotnbdp__s4mae_fine
  [ -f "$D/best.pt" ] || { echo "!! $D missing -- naming moved"; exit 1; }
  $PY -m rsna.train.predict "$D"
  echo "=== M fold $F done $(date -u) ==="
done

echo "H5 FOLDS COMPLETE $(date -u)"
$PY -m rsna.train.s5_gate
