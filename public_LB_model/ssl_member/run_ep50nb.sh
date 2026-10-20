#!/bin/bash
# Build the native-neighbour cache, then ep50nb folds 0/4, then the gate.
# Registered 2026-09-13_native_neighbour_channels.md
set -euo pipefail
cd /import/fox/will/graduation_thesis/RSNA_kaggle
PY=~/anaconda3/envs/rsna/bin/python
echo "=== build _nb cache (train) start $(date -u) ==="
$PY -m rsna.data.build_slot_cache --native-nb --workers 64
N=$(ls cache_slots_336_130mm_12sl_nb/train/*.pt | wc -l)
echo "built $N train studies"
if [ "$N" -ne 4407 ]; then echo "!! expected 4407 studies in the _nb cache, got $N"; exit 1; fi
$PY -m rsna.data.build_slot_cache --native-nb --split test --workers 8
echo "=== build done $(date -u) ==="
for F in 0 4; do
  echo "=== ep50nb fold ${F} start $(date -u) ==="
  $PY -m rsna.train.loop --fold $F --run-tag ep50nb --native-nb True --accum 1 --backbone vit_small_patch14_dinov2.lvd142m --band-hi 0.8 --band-lo 0.2 --batch-size 16 --cache slot --crop-mm 130.0 --ema-decay 0.999 --encoder-lr-mult 0.027 --epochs 50 --fine-attn True --freeze-epochs 0 --head-type hier --img-size 336 --labels '' --loss both --lr 0.0003 --mixup none --mixup-alpha 0.3 --n-series 6 --n-slices 4 --no-augment False --overfit 0 --policy fused --rank-macro False --rank-tau 1.0 --rank-trim 0.0 --rank-trim-warmup 5 --rank-w 1.0 --seed 0 --select-on tier1 --swa-epochs 5 --target y_ --val-policy fused --warmup-frac 0.05 --weight-decay 0.05 --workers 16
  D=artifacts/runs/f${F}_vit_small_patch14_dinov2_lvd142m__slotnb__ep50nb_fine
  if [ ! -f "$D/best.pt" ]; then echo "!! $D missing -- naming moved"; exit 1; fi
  $PY -m rsna.train.predict "$D"
  echo "=== ep50nb fold ${F} done $(date -u) ==="
done
echo "EP50NB PILOT COMPLETE $(date -u)"
$PY -m rsna.train.ep50nb_gate
