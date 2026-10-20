# Projects a parsed ReportExtraction into (target, weight) pairs. 
# Kept separate from the driver on purpose: re-projecting is free

from typing import Optional
from .schema import Status, Severity, Temporality, Finding, COLUMN_MAP

POSITIVE, NEGATIVE, MASK = 1.0, 0.0, None

# Findings that reports comment on only when present. Measured `not_mentioned`
# rates on the 103-report pilot: Synovitis 84%, Fracture 58%, Contusion 54%,
# Lateral OA 53%, Baker's 50%, PF OA 36%, Medial OA 35%  -- versus <=14% for
# everything else. For these, silence is evidence, not absence of evidence.
# Added after the diagnose.py
GROUP_B = {"Synovitis", "Fracture", "Contusion", "Lateral OA",
           "Baker's", "Medial OA", "PF OA"}

# P(gold=1 | not_mentioned) measured at 0.12 on the 12 gold dev studies.
# Down-weighted because that 12% is real label noise, not certainty.
NM_SOFT_Y, NM_SOFT_W = 0.1, 0.5
# Full 4x5 decision table. Every cell is listed deliberately; there is no
# fallthrough, so an unhandled combination raises instead of silently masking.
_TABLE: dict[tuple[Status, Optional[Temporality]], Optional[float]] = {}
for _t in list(Temporality) + [None]:
    _TABLE[(Status.uncertain, _t)] = MASK
    _TABLE[(Status.not_mentioned, _t)] = MASK
    _TABLE[(Status.present, _t)] = MASK
    _TABLE[(Status.absent, _t)] = MASK

# The only two cells that produce a training signal 
_TABLE[(Status.present, Temporality.current)] = POSITIVE
_TABLE[(Status.absent, Temporality.current)] = NEGATIVE

# Active graft/residual tear, which is a real current injury 
# reported under postoperative anatomy 
_TABLE[(Status.present, Temporality.postoperative)] = POSITIVE

# Optional graded variant, off by default. Only consulted for POSITIVE cells.
_SEVERITY_VALUE = {
    Severity.mild: 0.60,
    Severity.moderate: 0.80,
    Severity.severe: 0.95,
    Severity.unspecified: 0.80,
    None: 0.80,
}

def project_finding(f: Finding, column: str, graded: bool = False,
                    nm_policy: str = "soft_neg") -> tuple[float, float]:
    """Return (y, w). w == 0.0 means masked; y is then arbitrary (0.0).
    nm_policy: "soft_neg" (default) or "mask" for the ablation arm."""
    key = (f.status, f.temporality)
    if key not in _TABLE:
        raise ValueError(f"undefined projection cell: {key}")
    out = _TABLE[key]
    if out is MASK:
        if (f.status is Status.not_mentioned
                and nm_policy == "soft_neg" and column in GROUP_B):
            return NM_SOFT_Y, NM_SOFT_W
        return 0.0, 0.0
    if out == POSITIVE and graded:
        return _SEVERITY_VALUE[f.severity], 1.0
    return out, 1.0


def project_extraction(ex, graded: bool = False,
                       nm_policy: str = "soft_neg") -> dict[str, tuple[float, float]]:
    """Keyed by train.csv column name."""
    return {
        COLUMN_MAP[k]: project_finding(getattr(ex, k), COLUMN_MAP[k],
                                       graded=graded, nm_policy=nm_policy)
        for k in COLUMN_MAP
    }

def project_row(status: str, temporality: str, column: str,
                severity: str | None = None, graded: bool = False,
                nm_policy: str = "soft_neg") -> tuple[float, float]:
    """String-keyed entry point for re-projecting saved parquet rows.
    Guarantees diagnose.py and training see the identical projection."""
    return project_finding(
        Finding(status=Status(status), temporality=Temporality(temporality),
                severity=Severity(severity) if severity else None, evidence=""),
        column, graded=graded, nm_policy=nm_policy)
