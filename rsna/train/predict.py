#!/usr/bin/env python
# rsna/train/predict.py  --  dump OOF predictions from a saved checkpoint
#   python rsna/train/predict.py artifacts/runs/f0_convnext_tiny_fb_in1k
#
# Runs are selected on DECORRELATED ERRORS, not top AUC (plan, Phase 4), and a
# scalar T1 cannot express that. This writes the per-study predictions that make
# correlation measurable -- and that Phase 5 ensembling needs regardless.

from __future__ import annotations

import argparse, json, sys
from dataclasses import fields
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rsna.labels.schema import COLUMN_MAP
from rsna.train.loop import Cfg, make_loaders
from rsna.train.model import load_for_inference

LABELS = list(COLUMN_MAP.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--overwrite", action="store_true")
    # The registration asks for the terminal checkpoint reported SEPARATELY.
    # `best.pt` was the only reachable one, so that row was unsatisfiable.
    # NOTE when reporting last.pt: 25-epoch runs end `raw` and 50-epoch runs end
    # `ema`, which is what invalidated the earlier "last vs last" diagnostic.
    # Say which state a terminal number is in, or do not report it.
    ap.add_argument("--ckpt", default="best.pt",
                    help="checkpoint inside run_dir (e.g. last.pt, swa5.pt)")
    ap.add_argument("--out", default=None,
                    help="output filename inside run_dir; defaults to oof.parquet "
                         "for best.pt and oof_<ckpt>.parquet otherwise")
    a = ap.parse_args()
    run = Path(a.run_dir)
    out = run / (a.out or ("oof.parquet" if a.ckpt == "best.pt"
                           else f"oof_{Path(a.ckpt).stem}.parquet"))
    if out.exists() and not a.overwrite:
        print(f"skip (exists) {out}"); return

    # tolerate configs written before/after Cfg gained fields
    known = {f.name for f in fields(Cfg)}
    raw = json.loads((run / "config.json").read_text())
    cfg = Cfg(**{k: v for k, v in raw.items() if k in known})

    _, va, _, n_va = make_loaders(cfg)
    # load_for_inference, not KneeModel(pretrained=False): the latter omits
    # img_size (fatal for a ViT trained at 336) and picks the wrong fc_norm
    # variant, since that fallback only fires when PRETRAINED weights fail.
    # fine_attn and curia_config MUST be passed from the run's own config.
    # Neither can be discovered by trying variants: fine_attn adds no parameters,
    # so a fine_attn checkpoint strict-loads into a fine_attn=False model and
    # silently computes something else. This is the same class of bug as fc_norm.
    curia_cfg = None
    if cfg.backbone.startswith("curia"):
        from rsna.train.curia import cached_config_path
        curia_cfg = str(cached_config_path())
    m = load_for_inference(cfg.backbone, cfg.img_size, run / a.ckpt, "cuda",
                           fine_attn=cfg.fine_attn,
                           head_type=getattr(cfg, 'head_type', 'hier'),
                           n_regions=getattr(cfg, 'n_regions', 0),
                           curia_config=curia_cfg)

    rows = []
    with torch.no_grad():
        for b in va:
            lg = m(b["img"].cuda(non_blocking=True), b["series_type"].cuda(),
                   b["slice_pos"].cuda(), b["series_mask"].cuda()).float().cpu().numpy()
            for i, uid in enumerate(b["uid"]):
                rows.append(dict(
                    StudyInstanceUID=uid, fold=cfg.fold, run=run.name,
                    backbone=cfg.backbone,
                    **{f"p_{l}": float(lg[i, j]) for j, l in enumerate(LABELS)},
                    **{f"y_{l}": float(b["y"][i, j]) for j, l in enumerate(LABELS)},
                    **{f"w_{l}": float(b["w"][i, j]) for j, l in enumerate(LABELS)},
                    **{f"gold_{l}": float(b["gold"][i, j]) for j, l in enumerate(LABELS)}))
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    chk = pd.read_parquet(out)
    assert len(chk) == n_va and chk.StudyInstanceUID.is_unique, (len(chk), n_va)
    print(f"wrote {out}  {chk.shape}  ({cfg.backbone})")


if __name__ == "__main__":
    main()
