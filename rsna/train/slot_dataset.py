"""Dataset over the slot cache: cache_slots_336_130mm_12sl/<split>/<uid>.pt

The slot cache stores a DENSE (6, 12, 336, 336) uint8 array plus a (6,) presence
mask. This adapts it to the tensor contract `KneeModel` already speaks --
(S, k, 3, H, W) with a per-entry mask -- so the model needs no changes:

    slot          -> the S axis. Absent slots are DROPPED rather than passed as
                     zero images, so `collate_studies` pads them and the study
                     transformer masks them out of its attention. A zero image
                     would be a black knee the model has to learn to ignore.
    slot id 0-5   -> `series_type`. Same cardinality as the old six series
                     types, different semantics: these are the six medically
                     defined slots, not {plane} x {contrast}.
    12 slices     -> k 2.5D groups of 3 adjacent slices, drawn by the SAME
                     `_sample_indices` / `_stack_25d` the series cache uses, so
                     the two paths cannot drift in how a stack becomes channels.

Compared with the old `StudyDataset` at its shipped config this is CHEAPER per
study -- 6 slots x 4 groups at 336px is 2.7 Mpx against 6 series x 24 slices at
256px for 9.4 Mpx -- while resolving at 0.387 mm/px instead of 0.714.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rsna.data.geometry import SIDE_DEPENDENT_LABELS
from rsna.train.dataset import (IMAGENET_MEAN, IMAGENET_STD, LABELS,
                                _sample_indices, _stack_25d)

N_SLOT = 6


class SlotDataset(Dataset):
    """One study: (S<=6, k, 3, H, W) images, slot ids, depths, targets, weights."""

    def __init__(self, studies: list[str], labels: pd.DataFrame,
                 folds: pd.DataFrame, cache_dir: Path, *,
                 n_groups: int = 4, img_size: int = 336, train: bool = True,
                 target_prefix: str = "y_", transform=None, seed: int = 0,
                 label_subset: list[str] | None = None,
                 aux: dict | None = None):
        self.studies = list(studies)
        self.lab, self.folds = labels, folds
        self.cache = Path(cache_dir)
        self.n_groups, self.img_size = n_groups, img_size
        self.train, self.tp = train, target_prefix
        self.transform, self.seed = transform, seed
        # Auxiliary soft supervision (experiment B): uid -> (target, weight) for the
        # Synovitis column ONLY, on report-silent cells the main loss masks out. Kept
        # SEPARATE from y/w so the original loss terms are untouched; training only.
        self.aux = aux or {}
        self._syn_j = LABELS.index("Synovitis")
        # A SPECIALIST is made by zeroing the loss weight on every other label.
        # No architecture change: the head still emits 12 logits, but only these
        # receive gradient, so the encoder, both transformers and the slot
        # attention all specialise. The unused label queries simply go dead.
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
        n_slice = vol.shape[1]
        for k in range(N_SLOT):
            if mask[k] < 0.5:
                continue
            # Stratified in training, bin centres in eval -- identical to the
            # series path, so the group draw doubles as stack augmentation.
            idx = _sample_indices(n_slice, self.n_groups, self.train, rng)
            if vol.dim() == 5:
                # Native-neighbour cache (6,12,3,H,W): each centre's true source
                # neighbours are stored, so take the triplets directly. SAME
                # centre draw and RNG consumption as the default path -- only
                # what sits in channels 0 and 2 differs.
                x = vol[k][torch.from_numpy(idx)].float().div_(255.0)
            else:
                x = _stack_25d(vol[k], idx).float().div_(255.0)  # (k,3,336,336)
            if x.shape[-1] != self.img_size:
                x = torch.nn.functional.interpolate(
                    x, size=(self.img_size, self.img_size),
                    mode="bilinear", align_corners=False)
            if self.transform is not None:
                # ONE transform per slot: a per-group transform would misalign
                # the three channels of a 2.5D stack against each other.
                x = self.transform(x, rng)
            imgs.append(x.sub_(IMAGENET_MEAN).div_(IMAGENET_STD))
            slots.append(k)
            positions.append(idx / max(n_slice - 1, 1))

        row = self.lab.loc[uid]
        y = torch.tensor([row[f"{self.tp}{l}"] for l in LABELS], dtype=torch.float32)
        w = torch.tensor([row[f"w_{l}"] for l in LABELS], dtype=torch.float32)

        if self.keep is not None:
            w = w * torch.from_numpy(self.keep)

        # Laterality unresolved -> medial/lateral is unreliable for this study.
        # TRAINING ONLY: validation must score all 12, because the test set does.
        if self.train and bool(self.folds.loc[uid, "lat_ambiguous"]):
            for j, l in enumerate(LABELS):
                if l in SIDE_DEPENDENT_LABELS:
                    w[j] = 0.0

        y_aux = torch.zeros(len(LABELS), dtype=torch.float32)
        w_aux = torch.zeros(len(LABELS), dtype=torch.float32)
        if uid in self.aux:
            y_aux[self._syn_j], w_aux[self._syn_j] = self.aux[uid]
        return dict(uid=uid, y_aux=y_aux, w_aux=w_aux,
                    img=torch.stack(imgs),                              # (S,k,3,H,W)
                    series_type=torch.tensor(slots, dtype=torch.long),  # slot id
                    slice_pos=torch.tensor(np.stack(positions), dtype=torch.float32),
                    y=y, w=w,
                    gold=torch.tensor([row.get(f"gold_{l}", float("nan"))
                                       for l in LABELS], dtype=torch.float32))
