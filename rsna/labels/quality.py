"""How far can each weak label be trusted as a MEASUREMENT?

Tier 1 (agreement with the LLM labels, n~881/fold) is our promotion metric, and
it weights all twelve labels equally. But the labels are not equally good. Scored
against the 58 gold studies the fused label's own AUC ranges from **0.944** (ACL)
down to **0.676** (Synovitis) -- barely above a coin flip.

That is not a statement about the model. It is a statement about the RULER. On a
label whose target is near-random, a model can move Tier-1 AUC without moving
true skill, because part of what it fits is label noise. Two consequences, both
paid for:

  * submission B handed Synovitis and Fracture to a specialist on the strength of
    weak-OOF gains, and lost 0.003 LB. Ranking candidates by weak-OOF gain
    PREFERENTIALLY SELECTS the labels whose targets are least trustworthy,
    because those have the most room to move.
  * the MIL head ablation "failed" by -0.0099 macro, of which Synovitis alone
    was -0.0740. On the trustworthy labels it was a tie.

So report Tier 1 twice: all twelve for continuity with every run we have, and
restricted to label_q >= HQ_MIN as the metric to PROMOTE on.

These labels must still be PREDICTED -- they are in the competition's macro AUC.
This is about what we may steer on, not what we may ignore.

CAVEAT: label_q is itself estimated on n=58 (SE ~0.05), so a hard cut is coarse
and labels near the line can flip. `high_quality()` prints the set it chose.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from rsna.labels.schema import COLUMN_MAP

ROOT = Path(__file__).resolve().parents[2]
LABELS = list(COLUMN_MAP.values())
HQ_MIN = 0.85


def _auc(y, s):
    y = np.asarray(y)
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def label_quality(policy: str = "fused", refresh: bool = False) -> dict[str, float]:
    """AUC of each weak label against the gold labels, on the 58 gold studies."""
    cache = ROOT / f"artifacts/labels/label_quality_{policy}.json"
    if cache.is_file() and not refresh:
        return json.loads(cache.read_text())

    tr = pd.read_csv(ROOT / "data/train.csv",
                     dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    gold = tr[tr[LABELS].notna().all(axis=1)]
    lab = pd.read_parquet(ROOT / f"artifacts/labels/labels_llm_{policy}.parquet")
    if lab.index.name != "StudyInstanceUID":
        lab = lab.set_index("StudyInstanceUID")
    lab = lab.reindex(gold.index)

    q = {l: _auc(gold[l].astype(int).values, lab[f"y_{l}"].values) for l in LABELS}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(q, indent=2))
    return q


def high_quality(policy: str = "fused", hq_min: float = HQ_MIN,
                 verbose: bool = False) -> np.ndarray:
    """Boolean mask over LABELS: which ones weak-OOF may be steered on."""
    q = label_quality(policy)
    keep = np.array([q[l] >= hq_min for l in LABELS], bool)
    if verbose:
        print(f"label_q >= {hq_min} ({keep.sum()}/12, policy={policy}):")
        for l in sorted(LABELS, key=lambda x: -q[x]):
            print(f"  {'KEEP' if q[l] >= hq_min else '    '} {l:18s} {q[l]:.3f}")
    return keep


if __name__ == "__main__":
    high_quality(verbose=True)


def contamination_trim(policy: str = "fused", scale: float = 1.0,
                       cap: float = 0.5) -> dict[str, float]:
    """Per-label trimming fractions for rpAUC, from the measured contamination.

    WSAUC (Xie et al., TPAMI 2024) shows pairwise AUC ERM on contaminated labels
    stays CONSISTENT -- Corollary 3 gives R_PN~ = a R_PN + (1-a)/2 with
    a = 1 - etaP - etaN -- but its excess-risk bound scales as 1/a (Theorem 8).
    rpAUC (Def. 11) restores robustness by deleting the beta lowest-scoring
    positives and alpha highest-scoring negatives, which Prop. 12 shows are
    exactly the top-loss instances.

    **RETRACTED 2026-08-28 -- the derivation below is WRONG. Do not use.**
    The paper's etas are posteriors on the OBSERVED sets: theta_A = 1-etaP =
    P(Y=1|W=1), theta_B = etaN = P(Y=1|W=0), so a = P(Y=1|W=1) - P(Y=1|W=0) and
    the trimming is ASYMMETRIC: beta = P(Y=0|W=1), alpha = P(Y=1|W=0).
    `2*AUC-1` is Youden's J = P(W=1|Y=1) - P(W=1|Y=0), the likelihood-side
    quantity, which is PREVALENCE-BLIND. MCL reads J=0.805 (looks clean) but
    a=0.529 with 47.1% of its weak positives gold-negative; Effusion reads
    J=0.238 but a=0.686. The two disagree on which labels are contaminated.
    Corrected values are tabulated in the 2026-08-28_rpauc pre-registration.
    Even corrected, the observed-negative counts are 1 (Synovitis) to 13, so
    these cannot support twelve per-label rates. Kept only as a record.

    The superseded reasoning: a binary label's AUC against truth is
    1 - (etaP+etaN)/2, so a = 2*AUC - 1 and the symmetric noise rate is
    (1 - a)/2. Measured on the `w > 0` cells of the 58 gold studies:

        Synovitis a = -0.071   Effusion 0.238   Medial OA 0.496 ...  MCL 0.805

    Synovitis is BELOW ZERO, and Corollary 2's consistency proof divides by a --
    at a < 0 the inequality flips and ERM converges toward the WRONG ordering.
    That is the cleanest available explanation for its -0.0957 under the rank
    loss, and it is why trimming is aimed per label rather than uniformly.

    CAVEAT: estimated from 18-56 gold cells per label. Noisy. `cap` bounds how
    much of any label may be deleted.
    """
    tr = pd.read_csv(ROOT / "data/train.csv",
                     dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    gold = tr[tr[LABELS].notna().all(axis=1)]
    lab = pd.read_parquet(ROOT / f"artifacts/labels/labels_llm_{policy}.parquet")
    if lab.index.name != "StudyInstanceUID":
        lab = lab.set_index("StudyInstanceUID")
    lab = lab.reindex(gold.index)
    out = {}
    for l in LABELS:
        m = (lab[f"w_{l}"] > 0).values
        hard = _auc(gold[l].astype(int).values[m],
                    (lab[f"y_{l}"].values[m] > 0.5).astype(float))
        a = 2 * hard - 1 if np.isfinite(hard) else 1.0
        out[l] = float(np.clip(scale * (1.0 - a) / 2.0, 0.0, cap))
    return out
