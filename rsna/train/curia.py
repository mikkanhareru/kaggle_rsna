"""Curia-2: a DINOv2-base pretrained on radiology (MRI/CT), as a drop-in encoder.

Motivated by a measurement, not by novelty. Our DINOv2-S moves only 0.41% from
its initialisation during training, and what little it does move is patch_embed
(+1.93%) and pos_embed (+4.16%) -- the domain plumbing, learning to read MR
intensities and a 336px medical crop. Its semantic blocks barely change
(0.23-0.66%). So the encoder spends its whole adaptation budget arriving at the
domain and has nothing left for knee anatomy. A radiology-pretrained encoder
starts past both problems.

Two mismatches with our pipeline, and they are not symmetric:

  patch 16, not 14   Harmless. 336/16 = 21 exactly, so the slot cache is
                     UNCHANGED. (Our DINOv2-S needs patch 14; 336/14 = 24.)

  1 channel, not 3   Real. Curia was pretrained on single-channel images; our
                     input is a 2.5D stack of three ADJACENT SLICES carrying
                     through-plane context. Collapsing to one channel would
                     discard the reason `_stack_25d` exists. Instead the patch
                     embedding is inflated and rescaled -- see _inflate.

`num_features` reports the width AFTER pooling, because that is what KneeModel
uses to size `self.proj`.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

CURIA_REPO = "raidium/curia-2"
CONFIG_NAME = "curia_config.json"      # what bundle_slot ships for offline use


class CuriaEncoder(nn.Module):
    """HF Dinov2Model wrapped to present timm's encoder interface.

    KneeModel calls `encoder(x) -> (N, num_features)` and reads `.num_features`.
    HF returns a ModelOutput of per-token hidden states, so this adapts the
    calling convention rather than the model itself.
    """

    def __init__(self, repo: str = CURIA_REPO, in_chans: int = 3,
                 pool: str = "cls_mean", pretrained: bool = True,
                 config_path: str | Path | None = None):
        super().__init__()
        from transformers import Dinov2Config, Dinov2Model

        if pretrained:
            self.backbone = Dinov2Model.from_pretrained(repo)
        else:
            # Kaggle runs with NO INTERNET, and every weight is going to be
            # overwritten by our checkpoint anyway -- the fine-tuned state dict
            # contains the encoder. All that is needed here is the ARCHITECTURE,
            # so build it from a ~2 KB config shipped in the bundle.
            # `from_pretrained` would reach for the network and die.
            if config_path is None:
                raise ValueError("pretrained=False needs config_path "
                                 f"(the {CONFIG_NAME} shipped in the bundle)")
            self.backbone = Dinov2Model(
                Dinov2Config(**json.loads(Path(config_path).read_text())))

        cfg = self.backbone.config
        self.pool = pool
        self.num_features = cfg.hidden_size * (2 if pool == "cls_mean" else 1)
        # Inflate AFTER construction in both branches, so the pretrained and the
        # from-config models end up structurally identical. Doing it only on the
        # pretrained path is how the fc_norm bug happened: an architecture that
        # differs depending on how it was built, with nothing to signal it.
        if in_chans != cfg.num_channels:
            self._inflate(in_chans, cfg.num_channels)

    def _inflate(self, want: int, have: int) -> None:
        """Repeat the pretrained patch filters across channels and rescale.

        w_new[:, c] = mean_c(w_old) / want, for every c. For an input whose
        channels are identical this reproduces the pretrained response EXACTLY,
        because the sum over channels reconstructs the original filter. Our
        channels are adjacent slices -- nearly identical -- so the model starts
        at its pretrained behaviour and learns to exploit the differences from
        there. timm does this internally for in_chans mismatches; HF does not.
        """
        proj = self.backbone.embeddings.patch_embeddings.projection
        w = proj.weight.data                                   # (D, have, p, p)
        new = nn.Conv2d(want, proj.out_channels, proj.kernel_size,
                        proj.stride, proj.padding,
                        bias=proj.bias is not None)
        new.weight.data = w.mean(dim=1, keepdim=True).repeat(1, want, 1, 1) / want
        if proj.bias is not None:
            new.bias.data = proj.bias.data.clone()
        self.backbone.embeddings.patch_embeddings.projection = new
        self.backbone.config.num_channels = want
        self.backbone.embeddings.patch_embeddings.num_channels = want

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(pixel_values=x).last_hidden_state    # (N, 1+P, D)
        if self.pool == "cls_mean":
            return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=-1)
        return h[:, 0]


def cached_config_path(repo: str = CURIA_REPO) -> Path:
    """Local path to the repo's config.json, downloading it if absent (2 KB)."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, "config.json"))
