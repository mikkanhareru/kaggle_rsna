"""Header-only audit: can we recover the two acquisition axes, and does the
six-slot scheme cover the corpus?

Answers three questions before 36 GB is committed to disk:

  1. Are the identifying fields populated? Measured over 2,000 random series:
     SeriesDescription 81.8% usable, ScanOptions 85.5%, SequenceName 46.4%,
     TR/TE ~95%. Fat suppression is a PREPARATION, not a contrast property, so
     TR/TE cannot recover it -- it has to come from these fields or not at all.

  2. Do the recovered flags form two axes or one? The provided columns have
     Fluid_Sensitive == Fat_Suppression on all 24,371 rows, which is one axis
     wearing two names. Recovery is only worth doing if the 2x2 is not diagonal.

  3. Does a constant 130 mm crop fit? A crop wider than the image silently does
     nothing, which is the failure mode that leaves scale unnormalised with no
     error anywhere.

Reads one header per series. Writes nothing.

Run as a module, not as a path -- the package root has to be on sys.path for
`from rsna.data.geometry import ...` to resolve:

    python -m rsna.data.audit_slots
"""
from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

from rsna.data.geometry import plane_from_iop
from rsna.data.sequence import (DUMMY, fat_suppressed, usable_text,
                                weighting)

ROOT = Path(__file__).resolve().parents[2]
TAGS = ["SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence",
        "SequenceVariant", "InversionTime", "RepetitionTime", "EchoTime",
        "PixelSpacing", "Rows", "Columns", "ImageOrientationPatient"]

CROP_MM = 130.0


def _txt(ds, name):
    v = getattr(ds, name, None)
    if v is None:
        return ""
    if isinstance(v, (list, pydicom.multival.MultiValue)):
        return " ".join(map(str, v))
    return str(v)



def probe(args):
    study, series, d = args
    try:
        # The full listing, not just the first name: n_files is how the builder
        # breaks slot ties (thickest stack wins), so it has to be recorded here
        # rather than recomputed there from a second pass over the mount.
        files = os.listdir(d)
        ds = pydicom.dcmread(os.path.join(d, files[0]), stop_before_pixels=True,
                             force=True, specific_tags=TAGS)
        desc, seqn = _txt(ds, "SeriesDescription"), _txt(ds, "SequenceName")
        opts, scan = _txt(ds, "ScanOptions"), _txt(ds, "ScanningSequence")
        num = lambda n: (float(getattr(ds, n)) if getattr(ds, n, None) not in (None, "") else None)
        ps = [float(x) for x in ds.PixelSpacing] if getattr(ds, "PixelSpacing", None) else None
        iop = [float(x) for x in ds.ImageOrientationPatient] \
            if getattr(ds, "ImageOrientationPatient", None) else None
        tr, te, ti = num("RepetitionTime"), num("EchoTime"), num("InversionTime")
        blob, tokens, answerable = usable_text(desc, seqn, opts)
        return dict(
            StudyInstanceUID=study, SeriesInstanceUID=series, n_files=len(files),
            desc_ok=bool(desc.strip()) and not DUMMY.search(desc),
            seqn_ok=bool(seqn.strip()), opts_ok=bool(opts.strip()),
            weighting=weighting(blob, tr, te, scan),
            fatsat=fat_suppressed(blob, tokens, answerable, scan, ti),
            plane=plane_from_iop(np.array(iop)) if iop else None,
            rows=int(ds.Rows) if getattr(ds, "Rows", None) else None,
            cols=int(ds.Columns) if getattr(ds, "Columns", None) else None,
            row_sp=ps[0] if ps else None, col_sp=ps[1] if ps else None, error=None)
    except Exception as e:
        return dict(StudyInstanceUID=study, SeriesInstanceUID=series, error=str(e)[:60])


# name, plane, weighting predicate, fat-sat requirement
SLOTS = [
    ("SAG_FLUID_FS",   "Sagittal", ("PD", "T2"), True),
    ("COR_FLUID_FS",   "Coronal",  ("PD", "T2"), True),
    ("AX_FLUID_FS",    "Axial",    ("PD", "T2"), True),
    ("SAG_FLUID_NOFS", "Sagittal", ("PD", "T2"), False),
    # T1 is an explicit class, NOT "not fluid-sensitive". The negation also
    # admits GRE and unknown, which in a 2,000-series sample was 26.9% of
    # everything eligible for the bucket.
    ("COR_T1",         "Coronal",  ("T1",),      False),
    ("SAG_T1",         "Sagittal", ("T1",),      False),
]


def coverage(df, fs_col="fatsat", strict=True):
    """Fill counts per slot.

    `fs_col` selects the fat-suppression source. The recovered column is the
    audit's own three-state reading; `Fat_Suppression` is the provided column,
    which is what the BUILDER uses -- so the coverage that actually predicts the
    cache has to be computed under that policy, not under the recovered one.
    strict=True requires fat suppression to be KNOWN (recovered column only).
    """
    filled, per_study = Counter(), Counter()
    for _, g in df.groupby("StudyInstanceUID"):
        k = 0
        for name, plane, wts, fs in SLOTS:
            col = g[fs_col]
            sel = (g.plane == plane) & g.weighting.isin(wts)
            sel &= col.eq(True) if fs else (col.eq(False) if strict else col.ne(True))
            if sel.any():
                filled[name] += 1
                k += 1
        per_study[k] += 1
    return filled, per_study


def report_coverage(df, label, studies, **kw):
    filled, per_study = coverage(df, **kw)
    print(f"\n  -- {label}")
    for name, *_ in SLOTS:
        print(f"     {name:15s} {filled[name]:5d} / {studies}  "
              f"({100 * filled[name] / studies:5.1f}%)")
    print(f"     slots per study: {dict(sorted(per_study.items()))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--dump", action="store_true",
                    help="write artifacts/slot_headers_<split>.parquet for the "
                         "cache builder, so it reuses this header pass instead "
                         "of repeating 24,371 reads under a second definition")
    a = ap.parse_args()

    # <split>_series.csv is the authoritative list of raw series. series_meta is
    # a build artefact, and its `skipped` flag marks series whose cache file
    # already existed on a RESUMED build -- not series that are unusable. A
    # rebuild would therefore make an audit keyed off it silently omit valid
    # series, which is the exact class of failure this audit exists to catch.
    prov = pd.read_csv(ROOT / f"data/{a.split}_series.csv")
    jobs = [(r.StudyInstanceUID, r.SeriesInstanceUID,
             f"{ROOT}/data/{a.split}_series/{r.StudyInstanceUID}/{r.SeriesInstanceUID}")
            for r in prov.itertuples(index=False)]
    print(f"probing {len(jobs)} series ...")
    with ThreadPoolExecutor(32) as ex:
        df = pd.DataFrame(list(ex.map(probe, jobs)))
    bad = int(df.error.notna().sum())
    df = df[df.error.isna()].drop(columns="error")
    n, studies = len(df), df.StudyInstanceUID.nunique()
    print(f"  {n} series over {studies} studies, {bad} unreadable\n")

    print("=== 1. tag availability ===")
    for c in ("desc_ok", "seqn_ok", "opts_ok"):
        print(f"  {c:9s} {100 * df[c].mean():5.1f}%")

    print("\n=== 2. recovered weighting ===")
    print(df.weighting.value_counts().to_string())
    print("\n=== 2b. recovered fat suppression (three-state) ===")
    print(df.fatsat.map({True: "yes", False: "no", None: "unknown"})
            .fillna("unknown").value_counts().to_string())

    print("\n=== 3. do the two axes separate? (confidently classified only) ===")
    conf = df[df.fatsat.notna() & df.weighting.isin(["T1", "PD", "T2"])].copy()
    conf["fluid"] = conf.weighting.isin(["PD", "T2"])
    print(pd.crosstab(conf.fluid, conf.fatsat, rownames=["fluid_sensitive"],
                      colnames=["fat_suppressed"]).to_string())
    print("  -> a diagonal table means one axis, not two: recovery buys nothing.")

    m = conf.merge(prov[["SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression",
                         "Anatomical_Plane"]], on="SeriesInstanceUID", how="left")
    pf, pfs = m.Fluid_Sensitive.astype(bool), m.Fat_Suppression.astype(bool)
    print(f"\n  provided fluid == provided fatsat : {100 * (pf == pfs).mean():.1f}%  "
          f"(the degeneracy this whole exercise is meant to break)")
    print(f"  recovered fluid  vs provided fluid : {100 * (m.fluid == pf).mean():.1f}%")
    print(f"  recovered fatsat vs provided fatsat: {100 * (m.fatsat == pfs).mean():.1f}%")
    print(f"  derived plane    vs provided plane : {100 * (m.plane == m.Anatomical_Plane).mean():.1f}%")

    print("\n=== 4. six-slot coverage over studies ===")
    report_coverage(df, "recovered fat-sat, strict (unknown excluded)", studies,
                    fs_col="fatsat", strict=True)
    report_coverage(df, "recovered fat-sat, permissive (unknown = not-FS)", studies,
                    fs_col="fatsat", strict=False)
    # The one that matters: the builder takes fat suppression from the provided
    # column, not from header recovery. The two agree on 98.2% of series, so the
    # tables differ slightly -- and it is this one that predicts the cache.
    dfp = df.merge(prov[["SeriesInstanceUID", "Fat_Suppression"]],
                   on="SeriesInstanceUID", how="left")
    dfp["provided_fs"] = dfp.Fat_Suppression.astype(bool)
    report_coverage(dfp, "*** BUILDER POLICY: provided Fat_Suppression ***", studies,
                    fs_col="provided_fs", strict=True)
    stranded = df.weighting.isin(["GRE", "unknown"]).sum()
    print(f"\n  series eligible for no slot (GRE or unknown weighting): "
          f"{stranded} ({100 * stranded / n:.1f}%)")

    print("\n=== 5. does a 130 mm crop fit? ===")
    have = df.row_sp.notna() & df.col_sp.notna() & df.rows.notna() & df.cols.notna()
    print(f"  missing PixelSpacing/Rows/Columns : {int((~have).sum())} ({100 * (~have).mean():.2f}%)")
    h = df[have]
    fits = (CROP_MM / h.row_sp <= h.rows) & (CROP_MM / h.col_sp <= h.cols)
    print(f"  crop fits                         : {int(fits.sum())} ({100 * fits.mean():.2f}%)")
    print(f"  needs edge-padding                : {int((~fits).sum())} ({100 * (~fits).mean():.2f}%)")
    for ax, npx, sp in (("row", h.rows, h.row_sp), ("col", h.cols, h.col_sp)):
        print(f"  {ax} FOV mm p1/50/99: {np.round(np.percentile(npx * sp, [1, 50, 99]), 1)}")

    if a.dump:
        # Everything the builder needs to choose slots, and nothing it would
        # have to re-derive. Written per split so a train table can never be
        # silently consumed as a test one.
        out = ROOT / f"artifacts/slot_headers_{a.split}.parquet"
        df[["StudyInstanceUID", "SeriesInstanceUID", "n_files", "plane",
            "weighting", "fatsat", "rows", "cols", "row_sp", "col_sp"]] \
            .to_parquet(out, index=False)
        print(f"\nwrote {out}  ({len(df)} series)")


if __name__ == "__main__":
    main()
