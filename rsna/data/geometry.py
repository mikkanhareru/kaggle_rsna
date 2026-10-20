# DICOM patient space is LPS:  +x -> patient Left,  +y -> Posterior,  +z -> Superior
# ImageOrientationPatient = [row_dir(3), col_dir(3)]
#   row_dir = direction of increasing COLUMN index (image horizontal)
#   col_dir = direction of increasing ROW index    (image vertical)
# PixelSpacing = [row_spacing, col_spacing]  (between rows, between columns)

from __future__ import annotations

from dataclasses import dataclass
import numpy as np 

CANONICAL_SIDE = "R"          # all knees are mirrored to look like a right knee
AMBIGUOUS_MM = 20.0           # |center_x| below this is too close to midline to trust
IRREGULAR_TOL = 0.25          # fractional spacing deviation that flags a bad stack

# A wrong laterality mirrors medial and lateral. These four are the only labels
# it breaks -- the other eight are side-agnostic (patellofemoral is midline), so
# an ambiguous study should mask these rather than be dropped entirely.
# Measured: |center_x| < AMBIGUOUS_MM is 39.3% accurate and holds 17 of 28 total
# errors while being 2% of series; outside that band geometry is 99.2% accurate.
SIDE_DEPENDENT_LABELS = frozenset({
    "Medial OA", "Lateral OA", "Medial Meniscus", "Lateral Meniscus",
})

@dataclass(frozen=True)
class SliceHeader:
    '''one slice's geometry, straight from the DICOM header'''
    ipp: np.ndarray
    iop: np.ndarray
    rows: int
    cols: int
    pixel_spacing: tuple[float, float] #(row_spacing, col_spacing)
    
@dataclass(frozen=True)
class SeriesGeometry:
    plane: str                # Sagittal | Coronal | Axial
    normal: np.ndarray        # (3,) unit slice normal
    order: np.ndarray         # indices into the input list, in canonical order
    positions: np.ndarray     # sort-key value per slice, canonical order
    spacing: float            # median inter-slice spacing (mm)
    irregular: bool           # gaps inconsistent -> suspect stack
    center_x: float           # volume centre, LPS x (mm)
    laterality: str           # L | R
    lat_source: str           # tag | geometry | ambiguous
    flip_lr: bool             # mirror horizontally to reach CANONICAL_SIDE

# 1. Primitives
def slice_normal(iop: np.ndarray) -> np.ndarray:
    """Unit normal of the image plane."""
    n = np.cross(np.asarray(iop[:3], float), np.asarray(iop[3:], float))
    return n / np.linalg.norm(n)


def plane_from_iop(iop: np.ndarray) -> str:
    """Classify by the normal's dominant axis, never from SeriesDescription.

    An earlier note here claimed 180 of 250 sampled series read
    'DummySeriesDesc!'. That was a biased sample: measured over 2,000 random
    series the field is 12.6% dummy and 5.7% empty, so 81.8% is usable. The
    conclusion is unchanged -- 18% unreadable is reason enough, and the field
    does not reliably encode plane even when populated -- but the field is
    usable for the things IOP cannot give us, such as fat suppression.
    """
    n = np.abs(slice_normal(iop))
    return ("Sagittal", "Coronal", "Axial")[int(np.argmax(n))]


def slice_center(h: SliceHeader) -> np.ndarray:
    """Centre of the slice in patient coordinates.

    IPP is the centre of the FIRST voxel, i.e. a CORNER of the image. Using it
    directly for laterality is 84.7% accurate; walking to the centre first is
    96.2%. Measured on 1,064 tagged series -- see the Phase 2 notes.
    """
    row_dir, col_dir = np.asarray(h.iop[:3], float), np.asarray(h.iop[3:], float)
    row_sp, col_sp = h.pixel_spacing
    return (np.asarray(h.ipp, float)
            + row_dir * (h.cols - 1) / 2.0 * col_sp
            + col_dir * (h.rows - 1) / 2.0 * row_sp)

# 2. ordering 

def _canonical_axis(plane: str, normal: np.ndarray, laterality: str) -> float:
    """+1 to keep the natural sort order, -1 to reverse it.

    Fixes a consistent anatomical direction so the slice sequence means the same
    thing to the model in every study:
      Sagittal -> medial to lateral   (outward from midline)
      Coronal  -> anterior to posterior (+y in LPS)
      Axial    -> inferior to superior (+z in LPS)

    Takes the RESOLVED laterality, not center_x, so the DICOM tag governs the
    direction on the ~2% of tagged series where tag and geometry disagree.
    """
    if plane == "Sagittal":
        # lateral is away from the midline; a left knee sits at +x
        side = 1.0 if laterality == "L" else -1.0
        return float(np.sign(normal[0] * side) or 1.0)
    if plane == "Coronal":
        return float(np.sign(normal[1]) or 1.0)
    return float(np.sign(normal[2]) or 1.0)

def order_slices(headers: list[SliceHeader], plane: str,
                 normal: np.ndarray, laterality: str):
    """Sort by projection onto the slice normal, then orient canonically.

    InstanceNumber is NOT used: this dataset spans GE, Siemens, Philips, Toshiba
    and Canon, and its meaning is not consistent across them.
    """
    proj = np.array([float(np.dot(np.asarray(h.ipp, float), normal)) for h in headers])
    proj = proj * _canonical_axis(plane, normal, laterality)
    order = np.argsort(proj, kind="stable")
    pos = proj[order]

    diffs = np.diff(pos)
    if len(diffs) == 0:
        return order, pos, 0.0, False
    spacing = float(np.median(np.abs(diffs)))
    irregular = bool(
        spacing <= 0
        or np.any(np.abs(np.abs(diffs) - spacing) > IRREGULAR_TOL * spacing)
        or np.any(diffs <= 0)          # duplicate or non-monotonic positions
    )
    return order, pos, spacing, irregular

# 3. laterality 
def infer_laterality(center_x: float, tag: str | None) -> tuple[str, str]:
    """(laterality, source). The DICOM tag wins when present.

    Tag coverage is ~50% of studies. Geometry alone is 96.0% accurate per series
    and 95.9% by study-majority -- voting does not help, because the errors are
    correlated within a study (only 3.6% of studies have disagreeing series).
    """
    if tag in ("L", "R"):
        return tag, "tag"
    if abs(center_x) < AMBIGUOUS_MM:
        return ("L" if center_x > 0 else "R"), "ambiguous"
    return ("L" if center_x > 0 else "R"), "geometry"


def needs_lr_flip(plane: str, iop: np.ndarray, laterality: str) -> bool:
    """Should this image be mirrored horizontally to reach CANONICAL_SIDE?

    Handles laterality AND row-direction sign in one test, so series whose
    row_dir points the other way are normalised too.

    Takes the RESOLVED laterality rather than center_x: deriving the side here
    would silently overrule the DICOM tag on the ~2% of tagged series where the
    two disagree, mirroring them the wrong way despite an authoritative tag.

    Sagittal images are never mirrored: their in-plane axes are anterior-posterior
    and superior-inferior, so medial/lateral lives in the SLICE ORDER instead,
    which _canonical_axis already fixes. Mirroring a sagittal knee would flip
    anterior and posterior and destroy the ACL/PCL relationship.
    """
    if plane == "Sagittal":
        return False
    row_dir = np.asarray(iop[:3], float)
    side = 1.0 if laterality == "L" else -1.0       # +1 = left knee
    medial_dir_x = -side                            # medial points toward midline
    # canonical: medial on the LOW column side of the image
    return bool(row_dir[0] * medial_dir_x > 0) != (CANONICAL_SIDE == "L")

# 4. entery point 
def series_geometry(headers: list[SliceHeader],
                    laterality_tag: str | None = None) -> SeriesGeometry:
    """Everything the cache builder needs for one series."""
    if not headers:
        raise ValueError("empty series")
    iop = np.asarray(headers[0].iop, float)
    normal = slice_normal(iop)
    plane = plane_from_iop(iop)
    center_x = float(np.mean([slice_center(h)[0] for h in headers]))
    # resolve laterality FIRST -- both ordering and mirroring depend on it
    lat, src = infer_laterality(center_x, laterality_tag)
    order, pos, spacing, irregular = order_slices(headers, plane, normal, lat)
    return SeriesGeometry(
        plane=plane, normal=normal, order=order, positions=pos,
        spacing=spacing, irregular=irregular, center_x=center_x,
        laterality=lat, lat_source=src,
        flip_lr=needs_lr_flip(plane, iop, lat),
    )