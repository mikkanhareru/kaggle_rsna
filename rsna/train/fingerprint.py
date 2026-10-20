"""A portable identity for a trained model, checked before it is trusted.

Weights that load but are read through the wrong preprocessing produce
predictions, not errors. The submission is well formed, the log says nothing,
and the difference is a number no output of the run reveals. Scaling that never
happens, or happens twice, is enough on its own and changes no shape anywhere.

So a checkpoint carries the answer it gave to a question with no data in it. The
input is generated from a seed rather than read, so it is identical on any
machine, and it is pushed through the WHOLE forward path -- byte scaling,
ImageNet normalisation, the encoder, the slot attention, the label queries.
Any of those differing moves the output by order one; numerics differing between
two GPUs moves it by ~1e-5, which is why the tolerance sits between them.

This checks that the model computes what it computed when it was fitted. It
cannot check that the pixels reaching it are the right pixels -- the slot cache
manifest and `study_volume` answer to that.
"""
from __future__ import annotations

import numpy as np
import torch

from rsna.train.dataset import IMAGENET_MEAN, IMAGENET_STD

TOL = 2e-3
SEED = 2026


def synthetic_batch(img_size: int, n_slot: int = 6, k: int = 4, seed: int = SEED,
                    device="cpu"):
    """A fixed bag with no data in it. Batch entry 1 has a masked-out slot, so
    the masked branch of the study attention is exercised rather than skipped."""
    g = torch.Generator().manual_seed(seed)
    img = torch.randint(0, 256, (2, n_slot, k, 3, img_size, img_size),
                        generator=g, dtype=torch.uint8).float().div_(255.0)
    img = img.sub_(IMAGENET_MEAN).div_(IMAGENET_STD)
    stype = torch.arange(n_slot).unsqueeze(0).expand(2, -1).contiguous()
    spos = torch.linspace(0, 1, k).view(1, 1, k).expand(2, n_slot, -1).contiguous()
    smask = torch.ones(2, n_slot, dtype=torch.bool)
    smask[1, -1] = False
    return (img.to(device), stype.to(device), spos.to(device), smask.to(device))


@torch.no_grad()
def fingerprint(model, img_size: int, device="cpu", **kw) -> np.ndarray:
    """(2, 12) logits on the synthetic bag, in TRUE float32.

    Disabling autocast is not enough. `torch.backends.cudnn.allow_tf32` defaults
    to True, so convolutions on an Ampere-or-later card silently run at TF32's
    10-bit mantissa while matmuls (whose flag defaults to False) stay fp32. A
    conv-heavy model fingerprinted on an H100 and checked on a T4 -- which has no
    TF32 tensor cores at all -- then differs by ~1.6e-2, eight times the
    tolerance, with nothing wrong. A ViT does not, because it is almost all
    matmul; that asymmetry is exactly what made this look model-specific.

    With TF32 off the cross-device spread is ~3e-4, well inside TOL.
    """
    prev = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    was = model.training
    model.eval()
    try:
        out = model(*synthetic_batch(img_size, device=device, **kw)).float().cpu().numpy()
    finally:
        (torch.backends.cudnn.allow_tf32,
         torch.backends.cuda.matmul.allow_tf32) = prev
        if was:
            model.train()
    return out


def check(model, img_size: int, expected, device="cpu", tol: float = TOL, tag: str = ""):
    """Raise unless the model is the same map it was when fitted."""
    got = fingerprint(model, img_size, device)
    exp = np.asarray(expected, np.float32)
    if got.shape != exp.shape:
        raise RuntimeError(
            f"{tag}fingerprint shape {got.shape} != stored {exp.shape}: the "
            f"architecture is not the one these weights were fitted to")
    d = float(np.abs(got - exp).max())
    if d > tol:
        raise RuntimeError(
            f"{tag}fingerprint differs by {d:.4g} (tolerance {tol:g}). The weights "
            f"load but do not compute what they computed when fitted -- "
            f"preprocessing, resolution or architecture has moved between runs.")
    return d
