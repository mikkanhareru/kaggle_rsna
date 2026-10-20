"""Dataset over the MaxSpan cache: cache_maxspan_336_140mm_64sl/<split>/<uid>.pt

The point of this file is a CONTROLLED comparison. `KneeModel`, `loop.py`, the
folds, the augmentation and the budget are all unchanged; only where the pixels
come from differs. So a delta against the slot-cache runs measures the
REPRESENTATION and nothing else.

    slot cache      6 medically-defined slots x 12 slices, 130 mm, 20-80% span,
                    laterality canonicalised, longest series per slot
    maxspan cache   5 slots x {18,14,12,8,12} slices, 140 mm, 2-98% span,
                    no laterality fix, first matching series

Budget is deliberately matched: 6 slots x 4 groups = 24 crops there, 5 slots x 4
groups = 20 here. If MaxSpan wins it is not because we gave it more compute.

One deviation from the public pipeline, and it is the reason the cache stores a
`slot` array: their windows are cut over the flat 64-slice stack, so a window
centred at index 17 stacks sagittal-fluid slices 16 and 17 with sagittal-OTHER
slice 18 -- three channels from two pulse sequences. Here every window is cut
INSIDE one slot, because `_stack_25d` clamps at the ends of the array it is
handed. Whether that helps is an experiment; costing it nothing is not.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rsna.data.geometry import SIDE_DEPENDENT_LABELS
from rsna.data.maxspan import SLOTS, SLOT_BOUNDS
from rsna.train.dataset import (IMAGENET_MEAN, IMAGENET_STD, LABELS,
                                _sample_indices, _stack_25d)

N_SLOT = len(SLOTS)                      # 5


class MaxSpanDataset(Dataset):
    """One study: (S<=5, k, 3, H, W) images, slot ids, depths, targets, weights."""

    def __init__(self, studies: list[str], labels: pd.DataFrame,
                 folds: pd.DataFrame, cache_dir: Path, *,
                 n_groups: int = 4, img_size: int = 336, train: bool = True,
                 target_prefix: str = "y_", transform=None, seed: int = 0,
                 label_subset: list[str] | None = None):
        self.studies = list(studies)
        self.lab, self.folds = labels, folds
        self.cache = Path(cache_dir)
        self.n_groups, self.img_size = n_groups, img_size
        self.train, self.tp = train, target_prefix
        self.transform, self.seed = transform, seed
        self.keep = (None if not label_subset
                     else np.array([l in label_subset for l in LABELS], np.float32))

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, i: int) -> dict:
        uid = self.studies[i]
        rng = np.random.default_rng(None if self.train else (self.seed, i))
        d = torch.load(self.cache / f"{uid}.pt", map_location="cpu",
                       weights_only=False)
        vol, mask = d["vol"], d["mask"].numpy()

        imgs, slots, positions = [], [], []
        for k in range(N_SLOT):
            a, b = int(SLOT_BOUNDS[k]), int(SLOT_BOUNDS[k + 1])
            valid = np.where(mask[a:b] > 0)[0]
            if len(valid) == 0:
                continue                       # absent slot -> DROPPED, not zeroed
            # Trim to the readable span. A slot whose series was short leaves
            # black slices at the tail; sampling them would spend a quarter of
            # this slot's budget on an empty frame.
            lo, hi = a + int(valid.min()), a + int(valid.max()) + 1
            sub = vol[lo:hi]                    # (n, 336, 336) -- clamps here
            idx = _sample_indices(sub.shape[0], self.n_groups, self.train, rng)
            x = _stack_25d(sub, idx).float().div_(255.0)
            if x.shape[-1] != self.img_size:
                x = torch.nn.functional.interpolate(
                    x, size=(self.img_size, self.img_size),
                    mode="bilinear", align_corners=False)
            if self.transform is not None:
                x = self.transform(x, rng)      # one transform per slot
            imgs.append(x.sub_(IMAGENET_MEAN).div_(IMAGENET_STD))
            slots.append(k)
            positions.append(idx / max(sub.shape[0] - 1, 1))

        if not imgs:                            # no readable slot at all
            z = torch.zeros(self.n_groups, 3, self.img_size, self.img_size)
            imgs, slots = [z], [0]
            positions = [np.zeros(self.n_groups, np.float32)]

        row = self.lab.loc[uid]
        y = torch.tensor([row[f"{self.tp}{l}"] for l in LABELS], dtype=torch.float32)
        w = torch.tensor([row[f"w_{l}"] for l in LABELS], dtype=torch.float32)
        if self.keep is not None:
            w = w * torch.from_numpy(self.keep)

        # This cache does NOT canonicalise laterality, so medial/lateral is
        # unreliable for EVERY study, not only the ambiguous ones -- the model
        # has to read the side off the image itself. Kept as-is on purpose: the
        # public weights learned exactly this and reach 0.877 on Lateral
        # Meniscus, where our canonicalised pipeline reaches 0.778. Masking here
        # would test a different hypothesis than the one we are asking.
        if self.train and bool(self.folds.loc[uid, "lat_ambiguous"]):
            for j, l in enumerate(LABELS):
                if l in SIDE_DEPENDENT_LABELS:
                    w[j] = 0.0

        return dict(uid=uid,
                    img=torch.stack(imgs),
                    series_type=torch.tensor(slots, dtype=torch.long),
                    slice_pos=torch.tensor(np.stack(positions), dtype=torch.float32),
                    y=y, w=w,
                    gold=torch.tensor([row.get(f"gold_{l}", float("nan"))
                                       for l in LABELS], dtype=torch.float32))
