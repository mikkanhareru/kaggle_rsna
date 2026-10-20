"""Build label files from the public report labels, which are better than ours.

Measured against the 58 gold studies, our fused labels score macro **0.844**; the
public `llm_labels_v4_blend.csv` scores **0.893**, better on 10 of 12 labels. The
gains land precisely on the columns that have been distorting our decisions:

    Effusion   0.696 -> 0.877   (+0.181)
    Synovitis  0.676 -> 0.790   (+0.114)
    PF OA      0.819 -> 0.902   (+0.083)

That matters twice over. Labels are the SUPERVISION, so a better target should
give a better model. They are also the RULER: Tier 1 is agreement with them, so
with these the promotion metric covers ten or eleven labels instead of six.

Two policies are written, because the right one is an experiment, not a guess:

    public   their v4_blend alone      -- single variable against `fused`
    fused2   mean of ours and theirs   -- fusing at a flat 0.5 beat BOTH parents
                                          last time (0.8444 vs 0.8142/0.8352), and
                                          ours is better on Lateral Meniscus and
                                          Fracture, so there is something to keep

The `w_` mask and the `gold_`/`is_gold`/`lang` columns are copied from our own
file unchanged. Our masking is separately vindicated -- on the cells we discard,
the other reader's labels score AUC 0.541 -- so the mask is not up for revision
here; only the target values are.

    python -m rsna.labels.build_public
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rsna.labels.quality import label_quality
from rsna.labels.schema import COLUMN_MAP

ROOT = Path(__file__).resolve().parents[2]
LABELS = list(COLUMN_MAP.values())
SRC = ROOT / "data_more/rsna-knee-llm-report-labels/llm_labels_v4_blend.csv"


def main():
    base = pd.read_parquet(ROOT / "artifacts/labels/labels_llm_fused.parquet")
    if base.index.name != "StudyInstanceUID":
        base = base.set_index("StudyInstanceUID")
    pub = pd.read_csv(SRC, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    pub = pub.reindex(base.index)

    miss = pub[LABELS].isna().any(axis=1).sum()
    if miss:
        print(f"  {miss} studies missing from the public file -- keeping ours there")

    for policy, mix in (("public", 1.0), ("fused2", 0.5)):
        out = base.copy()
        for l in LABELS:
            theirs = pub[l].values.astype(float)
            ours = base[f"y_{l}"].values.astype(float)
            y = np.where(np.isnan(theirs), ours, mix * theirs + (1 - mix) * ours)
            out[f"y_{l}"] = np.clip(y, 0.0, 1.0)
        dst = ROOT / f"artifacts/labels/labels_llm_{policy}.parquet"
        out.to_parquet(dst)
        q = label_quality(policy, refresh=True)
        hq = [l for l in LABELS if q[l] >= 0.85]
        print(f"{policy:8s} -> {dst.name}   macro label_q {np.mean(list(q.values())):.4f}"
              f"   trustworthy labels {len(hq)}/12")

    q0 = label_quality("fused")
    print(f"\n{'label':18s} {'fused':>7s} {'public':>7s} {'fused2':>7s}")
    qs = {p: label_quality(p) for p in ("public", "fused2")}
    for l in LABELS:
        print(f"{l:18s} {q0[l]:7.3f} {qs['public'][l]:7.3f} {qs['fused2'][l]:7.3f}")
    for p in ("fused", "public", "fused2"):
        qq = label_quality(p)
        print(f"{p:8s} macro {np.mean(list(qq.values())):.4f}   "
              f"hq {sum(v >= 0.85 for v in qq.values())}/12")


if __name__ == "__main__":
    main()
