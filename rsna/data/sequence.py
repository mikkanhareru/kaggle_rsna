"""MR acquisition semantics from DICOM headers: weighting and fat suppression.

Used by the slot audit, the slot-cache builder and the Kaggle inference
notebook, so the three cannot drift apart. `preprocess.py` states the same
principle for the pixel path; this is its counterpart for the metadata path.

Two axes, recovered separately:

  weighting        T1 / PD / T2 / GRE, from the protocol name first and TR/TE
                   second. The name survives where TR/TE does not -- reading it
                   first cut `unknown` from 1,206 series to 1, and studies with
                   zero eligible slots from 238 to 0.

  fat suppression  a PREPARATION, not a contrast property, so TR/TE cannot see
                   it. It has to come from ScanOptions or the protocol name.

Measured over all 24,371 training series: SeriesDescription 82.3% usable,
ScanOptions 85.9%, SequenceName 45.9%, and the two axes genuinely separate
(5,110 fluid-sensitive non-fat-suppressed series against 13,653 suppressed).
"""
from __future__ import annotations

import re

# Bumped whenever a rule below changes the label it assigns. The cache manifest
# records it, so a cache built under one version refuses to resume under another
# rather than mixing two readings of the same corpus in one directory.
VERSION = 2

# Separators are folded BEFORE any regex runs. `_` is a word character, so
# \bfs\b does not match inside `pd_tse_fs_sag_320` -- one of the most common
# descriptions in this corpus. Folding first is what makes \b usable at all.
SEP = re.compile(r"[_\-.]+")
DUMMY = re.compile(r"dummy", re.I)

# Explicit weighting names. TIRM and STIR are inversion-recovery FLUID-sensitive
# sequences and belong with T2/PD; they must not read as T1 for being inversion
# based.
W_T1 = re.compile(r"\bt1\b|\bt1w\b|\bmprage\b|\bmp rage\b|\bt1 ?tse\b|\bt1 ?fse\b")
W_T2 = re.compile(r"\bt2\b|\bt2w\b|\bt2 ?star\b|\bstir\b|\btirm\b")
W_PD = re.compile(r"\bpd\b|\bpdw\b|\bdp\b|\bproton\b")

# Free-text fat-suppression cues. Deliberately NOT here: a bare `sat`, because a
# SAT option is usually a spatial saturation band, and a bare `dixon`, because
# Dixon reconstructs water-only, fat-only, in-phase and opposed-phase images and
# only the water-only one is suppressed.
FS_TEXT = re.compile(r"\bfs\b|\bfatsat\b|\bfat sat\b|\bstir\b|\btirm\b|"
                     r"\bspair\b|\bspir\b|\bfsat\b|\bchemsat\b|\bspecir\b")
FS_DIXON = re.compile(r"\bdixon\b.*\b(w|water)\b|\b(w|water)\b.*\bdixon\b")

# ScanOptions is a set of coded tokens, not prose. `SP FS PER` is a real value
# here: SP is spatial presaturation, FS is the fat saturation we want.
FATSAT_OPTS = {"FS", "FATSAT", "FAT_SAT", "FSAT"}

# STIR nulls fat at a short TI; a long-TI inversion (FLAIR, MP-RAGE) does not.
# 1.5 T STIR sits near 140-170 ms, 3 T near 200-250 ms.
STIR_TI = (80.0, 300.0)


def norm(s: str) -> str:
    return SEP.sub(" ", s.lower()).strip()


def usable_text(desc: str, seqn: str, opts: str):
    """(searchable blob, option tokens, whether anything could have answered).

    SeriesDescription and SequenceName are judged INDEPENDENTLY: a dummy
    description must not condemn a good sequence name beside it, nor contribute
    its own text to the blob.
    """
    d, s = norm(desc), norm(seqn)
    d_ok = bool(d) and not DUMMY.search(d)
    s_ok = bool(s)
    blob = " ".join(x for x, ok in ((d, d_ok), (s, s_ok)) if ok)
    # Split on whitespace and DICOM's backslash only. Folding `_` here would
    # break FAT_SAT into FAT and SAT and lose the match.
    tokens = {t.upper() for t in re.split(r"[\\\s]+", opts.strip()) if t}
    return blob, tokens, (d_ok or s_ok or bool(tokens))


def _weight_from_text(blob: str):
    """T1 / PD / T2 from an explicit protocol name, or None if it does not say.

    A name asserting both a T1 and a fluid-sensitive cue is contradictory and
    returns None so TR/TE can arbitrate. A name asserting both PD and T2 (a
    dual-echo acquisition) also returns None, but harmlessly: the two land in
    the same slot bucket either way.
    """
    if not blob:
        return None
    t1, t2, pd_ = bool(W_T1.search(blob)), bool(W_T2.search(blob)), bool(W_PD.search(blob))
    if t1 and not (t2 or pd_):
        return "T1"
    if t2 and not (t1 or pd_):
        return "T2"
    if pd_ and not (t1 or t2):
        return "PD"
    return None


def weighting(blob: str, tr, te, seq: str) -> str:
    """T1 / PD / T2 / GRE / unknown, protocol name first, TR/TE second.

    The name outranks ScanningSequence: `t1_fl2d` is a gradient echo AND a T1,
    and for slot assignment the T1 is what matters. The cost is that a `t2_ffe`
    files as T2 rather than GRE -- acceptable, since a T2*-weighted gradient echo
    does behave as a fluid-sensitive series.

    GRE remains its own class for anything the name does not settle: it is
    neither T1 nor fluid-sensitive, and folding it into T1 by negation is what
    contaminates a T1 slot (26.9% of the nominally eligible pool).
    """
    named = _weight_from_text(blob)
    if named is not None:
        return named
    if "GR" in seq:
        return "GRE"
    if tr is None or te is None:
        return "unknown"
    if tr < 800:
        return "T1"
    return "T2" if te >= 60 else "PD"


def fat_suppressed(blob: str, tokens: set, answerable: bool, scan: str, ti):
    """True / False / None. None means the fields could not answer.

    Absence of a cue is only evidence of absence when a cue could have been
    seen. Collapsing unknown into False is what falsely fills the NOFS and T1
    slots and makes the presence mask assert a sequence nobody acquired.

    Note: the slot-cache builder does not use this. Header recovery agrees with
    the provided Fat_Suppression column on 98.2% of series, and that column is
    equally available at inference without a header parse, so it is preferred
    there. This stays as the independent check that established the agreement.
    """
    if tokens & FATSAT_OPTS:
        return True
    if FS_TEXT.search(blob) or FS_DIXON.search(blob):
        return True
    # An inversion-recovery scan is STIR only at a fat-nulling TI. `IR` alone
    # also covers FLAIR and MP-RAGE, which suppress nothing of the sort.
    if "IR" in scan and ti is not None and STIR_TI[0] <= ti <= STIR_TI[1]:
        return True
    return False if answerable else None
