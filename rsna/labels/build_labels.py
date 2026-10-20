#!/usr/bin/env python
# ── Phase 1 / Block 6 ──────────────────────────────────────────────────
# findings_<tag>.parquet (52,884 long rows) -> labels_llm_<policy>.parquet (4,407 wide)
# This file is what Phase 2 onward consumes.
#
#   python rsna/labels/build_labels.py --tag full --nm-policy mask       # baseline
#   python rsna/labels/build_labels.py --tag full --nm-policy soft_neg   # challenger

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rsna.labels.project import project_row
from rsna.labels.schema import COLUMN_MAP

LABELS = list(COLUMN_MAP.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    # `mask` is the baseline: it makes no assumption about what silence means.
    # `soft_neg` must earn its extra coverage on downstream OOF AUC.
    ap.add_argument("--nm-policy", default="mask", choices=["mask", "soft_neg"])
    ap.add_argument("--no-agreement-weight", dest="agreement_weight",
                    action="store_false", default=True)
    a = ap.parse_args()

    f = pd.read_parquet(ROOT / f"artifacts/labels/findings_{a.tag}.parquet")

    def yw(r, graded):
        sev = r.severity if pd.notna(r.severity) else None
        return project_row(r.status, r.temporality, r.column,
                           severity=sev, graded=graded, nm_policy=a.nm_policy)

    f["y_bin"] = f.apply(lambda r: yw(r, False)[0], axis=1)
    f["y_graded"] = f.apply(lambda r: yw(r, True)[0], axis=1)

    # consensus_ok zeroes three-way ties rather than picking one arbitrarily.
    # Agreement down-weighting is a PRIOR on self-consistency, deliberately NOT
    # fitted from gold (which is spent on the one-time 0.865 evaluation).
    # It is an ablation arm for Phase 4, hence --no-agreement-weight.
    base = f.apply(lambda r: yw(r, False)[1], axis=1)
    f["w"] = base * f.consensus_ok * (f.agreement if a.agreement_weight else 1.0)

    wide = pd.concat(
        [f.pivot(index="StudyInstanceUID", columns="column", values=c)
          .reindex(columns=LABELS).add_prefix(p)
         for c, p in [("y_bin", "y_"), ("y_graded", "g_"), ("w", "w_")]], axis=1)

    # Gold labels are kept SEPARATE, never merged into y_/w_. Phase 4 folds on
    # these studies, so training must be able to drop a fold's gold rows -- and
    # their report-derived labels -- without unpicking a baked-in override.
    tr = pd.read_csv(ROOT / "data/train.csv").set_index("StudyInstanceUID")
    wide = wide.join(tr[LABELS].add_prefix("gold_"))
    wide["is_gold"] = wide[[f"gold_{l}" for l in LABELS]].notna().all(axis=1)
    wide["lang"] = (f.drop_duplicates("StudyInstanceUID")
                     .set_index("StudyInstanceUID").lang_llm)

    out = ROOT / f"artifacts/labels/labels_llm_{a.nm_policy}.parquet"
    wide.to_parquet(out)

    check = pd.read_parquet(out)          # readback in THIS environment
    assert check.shape == (4407, 50), check.shape
    assert check.index.is_unique
    print(f"wrote {out}\n  readback OK {check.shape} | gold={int(wide.is_gold.sum())} "
          f"| nm_policy={a.nm_policy} | agreement_weight={a.agreement_weight}")

    s = pd.DataFrame({
        "usable%": [(wide[f"w_{l}"] > 0).mean() * 100 for l in LABELS],
        "pos_rate": [wide.loc[wide[f"w_{l}"] > 0, f"y_{l}"].gt(0.5).mean() for l in LABELS],
        "mean_w": [wide.loc[wide[f"w_{l}"] > 0, f"w_{l}"].mean() for l in LABELS],
    }, index=LABELS).round(3)
    print(s.sort_values("usable%").to_string())


if __name__ == "__main__":
    main()
