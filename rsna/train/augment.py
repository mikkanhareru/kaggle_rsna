# Interface matches dataset.py:  transform(x, rng) -> x   with x (k,3,H,W) in [0,1]
# ONE transform per series: the k slices are neighbouring anatomy and the 3
# channels ARE neighbouring slices, so transforming them independently would
# misalign the very structure the 2.5D stack exists to capture.

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from monai.transforms import RandBiasField, RandGibbsNoise
from torchvision.transforms import v2


@dataclass
class AugConfig:
    # geometric -- NO horizontal flip, ever (see assert_no_hflip below)
    p_affine: float = 0.8
    degrees: float = 10.0
    translate: float = 0.08
    scale: tuple[float, float] = (0.90, 1.10)
    p_crop: float = 0.5
    crop_scale: tuple[float, float] = (0.80, 1.00)
    # intensity -- grayscale, not RGB ColorJitter
    p_intensity: float = 0.7
    brightness: float = 0.15
    contrast: float = 0.15
    gamma: tuple[float, float] = (0.80, 1.25)
    # MRI-specific (MONAI); albumentations has no equivalent
    p_bias: float = 0.30
    bias_coeff: float = 0.30
    p_gibbs: float = 0.10
    # misc
    p_noise: float = 0.30
    noise_std: float = 0.02
    p_erase: float = 0.25
    erase_frac: tuple[float, float] = (0.02, 0.12)
    p_slice_drop: float = 0.10
    slice_drop_frac: float = 0.10


class SeriesAugment:
    """Augments one series as a unit."""

    def __init__(self, cfg: AugConfig | None = None):
        self.cfg = cfg or AugConfig()
        c = self.cfg
        self.affine = v2.RandomAffine(degrees=c.degrees,
                                      translate=(c.translate, c.translate),
                                      scale=c.scale,
                                      interpolation=v2.InterpolationMode.BILINEAR)
        self.bias = RandBiasField(prob=1.0, coeff_range=(0.0, c.bias_coeff))
        self.gibbs = RandGibbsNoise(prob=1.0, alpha=(0.0, 0.5))

    def __call__(self, x: torch.Tensor, rng) -> torch.Tensor:
        c = self.cfg
        k, ch, H, W = x.shape

        # ── geometric: one sampled transform for the whole series ──
        if rng.random() < c.p_affine:
            x = self.affine(x)
        if rng.random() < c.p_crop:
            s = rng.uniform(*c.crop_scale)
            h, w = max(8, int(H * s)), max(8, int(W * s))
            top = int(rng.integers(0, H - h + 1))
            left = int(rng.integers(0, W - w + 1))
            x = torch.nn.functional.interpolate(
                x[:, :, top:top + h, left:left + w], size=(H, W),
                mode="bilinear", align_corners=False)

        # ── intensity: per SERIES, since it simulates acquisition differences ──
        if rng.random() < c.p_intensity:
            x = x * (1.0 + rng.uniform(-c.contrast, c.contrast))
            x = x + rng.uniform(-c.brightness, c.brightness)
            x = x.clamp_(0, 1).pow_(float(rng.uniform(*c.gamma)))

        # ── MRI-specific: coil inhomogeneity and reconstruction artefacts ──
        # 6 manufacturers in this dataset, so bias field is a real domain shift,
        # not a synthetic nuisance. Applied across k*ch so one field covers the
        # whole series rather than flickering slice to slice.
        if rng.random() < c.p_bias:
            flat = x.reshape(k * ch, H, W)
            x = torch.as_tensor(self.bias(flat)).reshape(k, ch, H, W).float()
        if rng.random() < c.p_gibbs:
            flat = x.reshape(k * ch, H, W)
            x = torch.as_tensor(self.gibbs(flat)).reshape(k, ch, H, W).float()

        # ── noise: per slice, which is how scanner noise actually behaves ──
        if rng.random() < c.p_noise:
            x = x + torch.randn_like(x) * c.noise_std

        # ── occlusion: SAME box across slices, so it reads as a real occluder ──
        if rng.random() < c.p_erase:
            frac = rng.uniform(*c.erase_frac)
            eh, ew = int((H * H * frac) ** 0.5), int((W * W * frac) ** 0.5)
            if eh > 1 and ew > 1:
                top = int(rng.integers(0, H - eh))
                left = int(rng.integers(0, W - ew))
                x[:, :, top:top + eh, left:left + ew] = float(rng.random())

        # ── slice dropout: robustness to variable stack length (16-320) ──
        if rng.random() < c.p_slice_drop and k > 4:
            n = max(1, int(k * c.slice_drop_frac))
            x[torch.from_numpy(rng.choice(k, n, replace=False))] = 0.0

        return x.clamp_(0, 1)


def assert_no_hflip(aug: SeriesAugment) -> None:
    """Guard the one augmentation that would silently corrupt four labels.

    A horizontal flip undoes the canonicalisation the whole of Phase 2 exists
    to establish, swapping medial and lateral. It costs nothing to make the
    prohibition executable rather than a comment.
    """
    names = [type(t).__name__ for t in vars(aug).values()]
    assert not any("Flip" in n or "Mirror" in n for n in names), \
        "horizontal flip would swap medial<->lateral; see geometry.CANONICAL_SIDE"

# ---------------------------------------------------------------------------
# Leave-one-FAMILY-out ablation. Four families, not nine transforms: nine tests
# at a +0.002 bar is a near-certain false positive, and the members of each
# family share a mechanism, so a family is the smallest unit worth a GPU-hour.
# Registered 2026-09-12. A family whose removal helps still needs a follow-up to
# identify WHICH transform in it was responsible -- removal is not attribution.
FAMILIES: dict[str, tuple[str, ...]] = {
    "geom":      ("p_affine", "p_crop"),
    "intensity": ("p_intensity",),          # brightness/contrast/gamma, one gate
    "mri":       ("p_bias", "p_gibbs"),     # MONAI coil/recon artefact simulation
    # p_slice_drop is listed for COMPLETENESS and is inert at n_slices=4: its
    # implementation is gated on `k > 4`. Removing this family at k=4 therefore
    # tests noise + erasing only, which is what the registration says it tests.
    "corrupt":   ("p_noise", "p_erase", "p_slice_drop"),
}

# Every probability gate must belong to exactly one family, or an "ablate
# everything, one family at a time" sweep silently leaves a transform untested
# in all four arms. Checked at import so it cannot drift as AugConfig grows.
_ALL_P = {f for f in AugConfig.__dataclass_fields__ if f.startswith("p_")}
_COVERED = {f for fs in FAMILIES.values() for f in fs}
assert _COVERED == _ALL_P, (
    f"family map does not partition the probability gates: "
    f"uncovered {_ALL_P - _COVERED}, unknown {_COVERED - _ALL_P}")
assert sum(len(v) for v in FAMILIES.values()) == len(_COVERED), \
    "a gate appears in more than one family; removals would not be disjoint"


def config_without(family: str) -> AugConfig:
    """AugConfig with one family's probability gates set to zero.

    "" returns the full stack unchanged, so the control arm and the ablation
    arms are built through the SAME code path -- a control built by a different
    route is a second recipe, not a control.
    """
    cfg = AugConfig()
    if not family:
        return cfg
    if family not in FAMILIES:
        raise SystemExit(f"unknown aug family {family!r}; choose from "
                         f"{sorted(FAMILIES)} or '' for the full stack")
    for f in FAMILIES[family]:
        setattr(cfg, f, 0.0)
    return cfg
