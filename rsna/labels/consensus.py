"""Self-consistency consensus over n sampled extractions.

Dependency-free (schema only) so reconsensus.py and analysis scripts can import
it without pulling in vLLM, which lives only in the rsna_llm env.
"""
from collections import Counter

from rsna.labels.schema import COLUMN_MAP, ReportExtraction


def consensus(samples: list[ReportExtraction]) -> dict:
    """Majority vote on the (status, temporality) PAIR.

    The projection depends on both fields, so a unanimous status with a split
    temporality is NOT a consensus -- it silently randomises present+current
    (POSITIVE) against present+historical (MASK).

    agreement    = the winning pair's share of samples
    consensus_ok = False when no pair wins more than once (a three-way tie),
                   which masks the row rather than picking one arbitrarily.
                   The n == 1 escape keeps the greedy pilot path valid.
    """
    n, out = len(samples), {}
    for key in COLUMN_MAP:
        fs = [getattr(s, key) for s in samples]
        (st, tmp), cnt = Counter(
            (f.status, f.temporality) for f in fs).most_common(1)[0]
        agree = [f for f in fs if (f.status, f.temporality) == (st, tmp)]
        sev = Counter(f.severity for f in agree).most_common(1)[0][0]
        out[key] = dict(
            status=st.value,
            severity=sev.value if sev else None,
            temporality=tmp.value,
            evidence=agree[0].evidence,
            agreement=cnt / n,
            consensus_ok=bool(cnt > 1 or n == 1),
        )
    return out
