# rsna/train/model.py
#
#   (B,S,k,3,H,W) study  ->  (B,12) logits
#
#   Stage A  per-slice encoder      timm backbone on 2.5D 3-channel stacks
#   Stage B  per-series aggregator  transformer over slices -> one series token
#   Stage C  study head             transformer over series + 12 label queries

from __future__ import annotations 

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

N_LABELS = 12
N_SERIES_TYPES = 6          # {Sag, Cor, Ax} x {T1/PD, fluid-sensitive}

class FourierPos(nn.Module):
    """Continuous slice depth in [0,1] -> D-dim embedding.

    Depth is continuous and stacks vary from 16 to 320 slices, so a learned
    lookup table over slice INDEX would not transfer between series lengths.
    Fourier features of the normalised position do.
    """
    def __init__(self, dim:int, n_freq: int=8):
        super().__init__()
        self.register_buffer("freq", 2.0 ** torch.arange(n_freq) * torch.pi)
        self.proj = nn.Linear(2 * n_freq, dim)
    
    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        a = pos.unsqueeze(-1) * self.freq
        return self.proj(torch.cat([a.sin(), a.cos()], dim=-1))
    
class AttenPool(nn.Module):
    """Leanred-query attention pooling over a token sequence."""
    def __init__(self, dim: int, heads: int=8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, key_padding_mask=None): # (B,T,D) -> (B,D)
        q = self.q.expand(x.shape[0], -1, -1)
        o, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask,
                         need_weights=False)
        return self.norm(o.squeeze(1))
    
def _encoder_layer(dim:int, heads:int, drop:float) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        dim, heads, dim_feedforward=dim * 2, dropout=drop,
        batch_first=True, norm_first=True, activation="gelu"
    )
    
class KneeModel(nn.Module):
    def __init__(self, backbone: str = "convnext_tiny.fb_in1k", *,
                 pretrained: bool = True, img_size: int | None = None,
                 dim: int = 384, heads: int = 8,
                 slice_layers: int = 2, study_layers: int = 2,
                 dropout: float = 0.1, drop_path: float = 0.1,
                 fc_norm: bool | None = None, fine_attn: bool = False,
                 head_type: str = "hier", n_regions: int = 0,
                 curia_config: str | None = None):
        super().__init__()
        
        if backbone.startswith("curia"):
            # NOT a timm model. The img_size probe and the fc_norm fallback
            # below are both timm-specific and would raise on a name timm does
            # not know, so this path exits before either of them.
            from rsna.train.curia import CuriaEncoder, cached_config_path
            self.encoder = CuriaEncoder(
                pretrained=pretrained,
                config_path=None if pretrained else curia_config)
            self.has_drop_path = False
        else:
            self._build_timm(backbone, pretrained, img_size, drop_path, fc_norm)
        enc_dim = self.encoder.num_features
        self.proj = nn.Linear(enc_dim, dim) if enc_dim != dim else nn.Identity()
        self._init_heads(dim, heads, slice_layers, study_layers, dropout,
                         fine_attn, head_type, n_regions)

    def _build_timm(self, backbone, pretrained, img_size, drop_path, fc_norm):
        # Not every timm family implements stochastic depth -- DenseNet raises
        # TypeError on drop_path_rate. Fall back rather than exclude the model.
        _kw = dict(pretrained=pretrained, in_chans=3, num_classes=0,
                   global_pool="avg")
        # Swin and ViT hard-assert their pretrained input size in patch_embed;
        # CNNs accept anything. Passing img_size makes timm interpolate the
        # relative-position bias so pretrained weights still load.
        if img_size is not None:
            try:
                timm.create_model(backbone, pretrained=False, img_size=img_size)
                _kw["img_size"] = img_size
            except TypeError:
                pass          # CNN family: no img_size arg, and none needed
        # timm's ViT renames its final LayerNorm `norm` -> `fc_norm` whenever
        # global_pool == "avg", so a DINOv2 checkpoint (which stores `norm`)
        # fails to load with "Missing key(s): fc_norm.weight". `fc_norm=False`
        # keeps the original name AND still mean-pools the patch tokens. It is
        # tried only on failure so no other family's behaviour changes.
        def _make(**extra):
            try:
                m = timm.create_model(backbone, drop_path_rate=drop_path,
                                      **_kw, **extra)
                return m, True
            except TypeError:
                return timm.create_model(backbone, **_kw, **extra), False

        # `fc_norm` may be forced. It has to be forceable, because the fallback
        # below only fires when PRETRAINED weights fail to load: with
        # pretrained=False it never fires, so inference would build fc_norm=True
        # while training built fc_norm=False and the state dict would not fit.
        # Same class name, same shapes elsewhere, different architecture.
        if fc_norm is not None:
            self.encoder, self.has_drop_path = _make(fc_norm=fc_norm)
        else:
            try:
                self.encoder, self.has_drop_path = _make()
            except RuntimeError as e:
                if "fc_norm" not in str(e):
                    raise
                self.encoder, self.has_drop_path = _make(fc_norm=False)

    def _init_heads(self, dim, heads, slice_layers, study_layers, dropout,
                    fine_attn, head_type="hier", n_regions=0):
        self.head_type = head_type
        self.n_regions = 0 if head_type in ("mil", "mean") else int(n_regions)
        self.drop = nn.Dropout(dropout)
        self.head_w = nn.Parameter(torch.randn(N_LABELS, dim) * 0.02)
        self.head_b = nn.Parameter(torch.zeros(N_LABELS))
        if head_type == "mean":
            # "No attention mechanisms, just simple pooling operations."
            # `mil` minus the learned attention: the ONLY aggregator here that
            # adds NO parameters beyond head_w/head_b, so a result is
            # attributable to the encoder and nothing else. Permutation
            # invariant, no slice position, no slot embedding -- and unlike
            # `mil`, nothing learned mixes the windows either.
            self.fine_attn = False       # meaningless without pooled slot tokens
            return
        if head_type == "mil":
            # The public 0.924 model's aggregator, verbatim in structure: score
            # every window independently, softmax over WINDOWS, take a per-label
            # weighted mean. Two Linear layers against our four transformer
            # stacks -- and PERMUTATION-INVARIANT, with no slice position and no
            # slot embedding. That absence is the hypothesis, so nothing
            # positional is added here; adding it would test something else.
            self.mil_norm = nn.LayerNorm(dim)
            self.mil_attn = nn.Sequential(
                nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(dropout),
                nn.Linear(256, N_LABELS))
            self.fine_attn = False       # meaningless without pooled slot tokens
            return
        self.slice_pos = FourierPos(dim)
        self.type_emb = nn.Embedding(N_SERIES_TYPES, dim)
        self.slice_tf = nn.TransformerEncoder(
            _encoder_layer(dim, heads, dropout), slice_layers,
            enable_nested_tensor=False)   # pre-norm disables the fast path anyway
        self.slice_pool = AttenPool(dim, heads)
        
        self.study_tf = nn.TransformerEncoder(
            _encoder_layer(dim, heads, dropout), study_layers,
            enable_nested_tensor=False)
        self.label_q = nn.Parameter(torch.randn(N_LABELS, dim) * 0.02)
        self.label_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.label_norm = nn.LayerNorm(dim)
        self.fine_attn = fine_attn
        if self.n_regions:
            # 2x2 spatial summaries per slice image, appended ONLY to the final
            # label-attention memory. Everything upstream -- slice_tf, slice_pool,
            # study_tf -- sees exactly what it saw before, so this does not
            # re-open the aggregation experiment. Putting regions into slice_tf
            # would take its sequence from k=4 to k*5=20 and its self-attention
            # matrix up 25x, which is a different experiment.
            #
            # NOT "additive": extra keys change the label-attention softmax
            # normalisation and its gradients. The old function is reachable, not
            # preserved. A null result therefore reads "no detected benefit from
            # this recipe", never "the regions went unused".
            self.region_emb = nn.Embedding(self.n_regions, dim)
        
     # two-stage training: warm the encoder, then freeze it and train aggregators
    def set_encoder_trainable(self, flag: bool) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = flag 
    
    def _encode_windows(self, flat, series_mask, k):
        """Encode only the VALID windows; scatter results back, padding as zeros.

        Padded windows are zeros and were previously encoded ALONGSIDE real ones.
        Any cross-sample normaliser inside the encoder then mixes them in before
        the mask is applied downstream -- and BatchNorm is exactly that. Measured
        BN module counts in the backbones this project uses:

            DINOv2 ViT-S      0     ConvNeXt-Tiny       0    (LayerNorm only)
            CoAtNet-rmlp-2   25     EfficientNetV2-S  110    ResNet34  36

        23.5% of the windows in a batch are padding (mean 4.588 of 6 slots valid;
        93.4% of studies carry at least one empty slot), so for a BN encoder about
        a quarter of every training batch statistic came from zeros.

        Scope of the change, measured:
          * LayerNorm-only encoders are per-sample -> BIT-IDENTICAL. The whole
            DINOv2 run history stays comparable, which the test asserts.
          * BN encoders change in TRAINING only. Eval-mode BN uses running
            statistics, so no already-computed inference output moves and no
            submitted prediction is affected.
        """
        keep = series_mask.repeat_interleave(k, dim=1).reshape(-1)   # slot-major
        idx = keep.nonzero(as_tuple=True)[0]
        if idx.numel() == flat.shape[0]:
            return self.encoder(flat)        # nothing padded: identical, and cheaper
        if idx.numel() == 0:
            # Cannot happen with this cache (every study has >= 3 valid slots) but
            # an empty batch makes BN raise, so never hand it one.
            return self.encoder(flat)
        f = self.encoder(flat[idx])
        out = f.new_zeros((flat.shape[0], f.shape[-1]))
        return out.index_copy(0, idx, f)

    def _encode(self, img, series_type, slice_pos, series_mask):
        """Pixels -> (slice tokens, slot tokens, pad mask, shapes).

        Shared by forward_embed and the attention diagnostic. They used to
        duplicate this whole path, which is exactly how the diagnostic came to
        report attention over a memory the model no longer used.
        """
        B, S, k = img.shape[:3]
        flat = img.reshape(B * S * k, *img.shape[3:])
        r = None
        if self.n_regions:
            # ONE encoder pass. forward_head(forward_features(x)) is bit-identical
            # to encoder(x) under BOTH norm layouts this project produces -- the
            # trained checkpoints carry norm=LayerNorm/fc_norm=Identity (the
            # pretrained-load fallback at _build_timm), while a default
            # pretrained=False build carries the reverse. Routing through the
            # encoder's own modules is what makes this layout-agnostic; forcing a
            # LayerNorm here would silently double-normalise the trained layout.
            # NOT packed: this branch needs the patch grid, and packing it is a
            # separate change. Safe only for a LayerNorm encoder -- assert that
            # rather than let a BN backbone silently take the leaky path.
            assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm)
                           for m in self.encoder.modules()), \
                "n_regions with a BatchNorm encoder would normalise over padded " \
                "windows; pack _encode's regional branch before using it"
            feats = self.encoder.forward_features(flat)           # (B*S*k, 1+P, enc)
            g = self.encoder.forward_head(feats)                  # == self.encoder(flat)
            sp = feats[:, self.encoder.num_prefix_tokens:]        # drop CLS/registers
            side = int(round(sp.shape[1] ** 0.5))
            assert side * side == sp.shape[1], \
                f"patch grid not square: {sp.shape[1]} tokens after {self.encoder.num_prefix_tokens} prefix"
            h = side // 2
            assert h * 2 == side, f"grid {side} not divisible into 2x2"
            gr = sp.reshape(-1, side, side, sp.shape[-1])
            r = torch.stack([gr[:, :h, :h], gr[:, :h, h:],
                             gr[:, h:, :h], gr[:, h:, h:]], 1).mean((2, 3))
            r = self.encoder.fc_norm(r)                           # no-op or LN, per layout
            r = self.proj(r).reshape(B, S, k, self.n_regions, -1)
        else:
            g = self._encode_windows(flat, series_mask, k)        # (B*S*k, enc)
        x = self.proj(g).reshape(B * S, k, -1)                    # (B*S, k, D)
        pos = self.slice_pos(slice_pos.reshape(B * S, k))
        typ = self.type_emb(series_type.reshape(B * S, 1))
        x = x + pos
        x = x + typ                                               # broadcast over k
        x = self.slice_tf(x)                                      # (B*S, k, D)
        s = self.slice_pool(x).reshape(B, S, -1)                  # (B, S, D)
        pad = ~series_mask
        s = self.study_tf(s, src_key_padding_mask=pad)            # cross-slot context
        if r is not None:
            # Same slice position and slot type the global token gets, plus a
            # region identity. No region embedding is added to global tokens.
            r = (r + pos.reshape(B, S, k, 1, -1)
                   + typ.reshape(B, S, 1, 1, -1)
                   + self.region_emb.weight.view(1, 1, 1, self.n_regions, -1)
                   + s.reshape(B, S, 1, 1, -1))                   # slot context, as `fine`
        return x, s, pad, B, S, k, r

    def _memory(self, x, s, pad, B, S, k, r=None):
        """What the label queries may attend over.

        `slice_pool` collapses k slices with a SINGLE learned query that does
        not know which finding is asking, so a Baker's-cyst query can pick the
        axial slot but not the posterior slices of it. With fine_attn the slice
        tokens are exposed alongside the pooled slot tokens -- strictly
        additive, so if per-slice detail is useless the attention can ignore it
        and recover the previous behaviour.
        """
        if not self.fine_attn and r is None:
            return s, pad
        parts, pads = [s], [pad]
        if self.fine_attn:
            fine = x.reshape(B, S, k, -1) + s.unsqueeze(2)        # (B,S,k,D)
            parts.append(fine.reshape(B, S * k, -1))
            # repeat_interleave, NOT repeat: `fine` is slot-major (s0k0..s0k3,
            # s1k0..), so the mask must be too. `repeat` would misalign every slot.
            pads.append(pad.repeat_interleave(k, dim=1))
        if r is not None:
            # r is (B,S,k,R): slot-major, then slice, then region. A padded slot
            # therefore owns k*R CONSECUTIVE entries. repeat_interleave(k*R) is
            # the only thing that lines up; `repeat` would tile the slot mask R
            # times and mask the wrong studies -- and with 96 region tokens that
            # failure trains and converges perfectly well while attending to the
            # wrong knees. test_regions.py checks this behaviourally, not by shape.
            R = r.shape[3]
            parts.append(r.reshape(B, S * k * R, -1))
            pads.append(pad.repeat_interleave(k * R, dim=1))
        mem, mem_pad = torch.cat(parts, dim=1), torch.cat(pads, dim=1)
        assert mem.shape[1] == mem_pad.shape[1], (mem.shape, mem_pad.shape)
        return mem, mem_pad

    def _forward_mil(self, img, series_mask):
        """Flat per-label MIL over all S*k windows -> (B, 12, D).

        No slice transformer, no slot transformer, no attention pool: the ONLY
        thing that mixes windows is the softmax. `head()` is then unchanged --
        their `clsW`/`clsb` are exactly our `head_w`/`head_b`.
        """
        B, S, k = img.shape[:3]
        x = self._encode_windows(img.reshape(B * S * k, *img.shape[3:]), series_mask, k)
        x = self.proj(x).reshape(B, S * k, -1)                    # (B, K, D)
        h = self.mil_norm(x)
        a = self.mil_attn(h)                                      # (B, K, 12)
        # repeat_interleave, NOT repeat: `x` is slot-major (s0k0..s0k3, s1k0..),
        # so the mask must be too. `repeat` would mask the wrong windows -- the
        # same trap `_memory` documents.
        pad = (~series_mask).repeat_interleave(k, dim=1)          # (B, K)
        # -1e4, not -inf: finite under fp16/bf16 autocast, and a study whose
        # every window is padded still yields a uniform softmax instead of NaN.
        a = a.masked_fill(pad.unsqueeze(-1), -1e4)
        o = torch.einsum("bkn,bkd->bnd", a.softmax(dim=1), h)     # (B, 12, D)
        return self.drop(o)

    def _forward_mean(self, img, series_mask):
        """Uniform mean over all S*k windows -> (B, 12, D).

        `_forward_mil` with the softmax replaced by a masked uniform average.
        The twelve labels share ONE pooled vector and differ only in
        head_w/head_b -- no per-label pooling at all, which is the point: this
        is the arm with no learned window mixing anywhere.
        """
        B, S, k = img.shape[:3]
        x = self._encode_windows(img.reshape(B * S * k, *img.shape[3:]), series_mask, k)
        x = self.proj(x).reshape(B, S * k, -1)                    # (B, K, D)
        # repeat_interleave, NOT repeat: `x` is slot-major (s0k0..s0k3, s1k0..),
        # so the mask must be too -- the same trap `_memory` documents.
        keep = series_mask.repeat_interleave(k, dim=1).unsqueeze(-1)   # (B,K,1)
        # Divide by the REAL window count, never K. Padded slots contribute
        # zero, so dividing by K would shrink every study that is missing a
        # series -- a silent per-study scale correlated with which series that
        # study happens to carry. clamp(min=1) keeps an all-padded study finite.
        n = keep.sum(1).clamp(min=1)                              # (B, 1)
        o = (x * keep).sum(1) / n                                 # (B, D)
        return self.drop(o.unsqueeze(1).expand(-1, N_LABELS, -1))  # (B, 12, D)

    def forward_embed(self, img, series_type, slice_pos, series_mask):
        """Everything up to the linear head -> (B, 12, D).

        Split out so manifold MixUp can mix HERE: (B,12,D) is shape-stable no
        matter how many series each study carries, unlike the pixels, where
        slot i of study A may be a sagittal T1 and slot i of B an axial PD-FS.
        """
        if self.head_type == "mean":
            return self._forward_mean(img, series_mask)
        if self.head_type == "mil":
            return self._forward_mil(img, series_mask)
        x, s, pad, B, S, k, r = self._encode(img, series_type, slice_pos, series_mask)
        mem, mem_pad = self._memory(x, s, pad, B, S, k, r)
        q = self.label_q.unsqueeze(0).expand(B, -1, -1)           # (B, 12, D)
        o, _ = self.label_attn(q, mem, mem, key_padding_mask=mem_pad,
                               need_weights=False)
        return self.drop(self.label_norm(o))                      # (B, 12, D)

    def head(self, o):
        """(B, 12, D) -> (B, 12) logits. One projection per label query."""
        return (o * self.head_w).sum(-1) + self.head_b

    def forward(self, img, series_type, slice_pos, series_mask):
        return self.head(self.forward_embed(img, series_type, slice_pos, series_mask))

    @torch.no_grad()
    def label_series_attention(self, img, series_type, slice_pos, series_mask):
        """Which slot each label attended to -> (B, 12, S). Diagnostic only.

        Under fine_attn the memory holds S slot tokens followed by S*k slice
        tokens, so a slot's total attention is its own token plus its slices.
        Reporting only the first S would understate every slot by whatever the
        fine tokens absorbed -- a plot of a computation that is not happening.
        """
        if self.head_type == "mean":
            # Uniform by construction. Returning zeros or the mil path would be
            # a plot of a computation that is not happening -- the same failure
            # this method's docstring warns about for fine_attn.
            raise NotImplementedError(
                "head_type='mean' pools uniformly; there is no attention to report")
        if self.head_type == "mil":
            return self._forward_mil(img, series_mask)
        x, s, pad, B, S, k, r = self._encode(img, series_type, slice_pos, series_mask)
        mem, mem_pad = self._memory(x, s, pad, B, S, k, r)
        q = self.label_q.unsqueeze(0).expand(B, -1, -1)
        _, w = self.label_attn(q, mem, mem, key_padding_mask=mem_pad,
                               need_weights=True, average_attn_weights=True)
        # Fold every appended token back onto its slot, or the plot describes a
        # computation that is not happening -- the same bug this docstring warns
        # about, which regions would silently re-introduce.
        out = w[:, :, :S]
        off = S
        if self.fine_attn:
            out = out + w[:, :, off:off + S * k].reshape(B, N_LABELS, S, k).sum(-1)
            off += S * k
        if r is not None:
            R = r.shape[3]
            out = out + w[:, :, off:off + S * k * R].reshape(
                B, N_LABELS, S, k * R).sum(-1)
        return out                                                   # (B, 12, S)


def build_model(cfg) -> KneeModel:
    """Unused today -- loop.py constructs KneeModel directly. Kept in step with
    it anyway: it silently dropped img_size and fine_attn, so the first caller
    to reach for it would have built a different model than training does."""
    return KneeModel(cfg.backbone, pretrained=cfg.pretrained, dim=cfg.dim,
                     img_size=cfg.img_size, fine_attn=cfg.fine_attn,
                     head_type=getattr(cfg, "head_type", "hier"),
                     n_regions=getattr(cfg, "n_regions", 0),
                     dropout=cfg.dropout, drop_path=cfg.drop_path)
        
                

        
    

def load_for_inference(backbone: str, img_size: int, ckpt, device="cpu",
                       fine_attn: bool = False, head_type: str = "hier",
                       n_regions: int = 0,
                       curia_config: str | None = None) -> KneeModel:
    """Build a KneeModel that actually accepts `ckpt`, and verify it did.

    The architecture a backbone name maps to is not a pure function of the name:
    timm's ViT calls its final LayerNorm `norm` or `fc_norm` depending on a flag
    that, during training, was chosen by a fallback which only triggers when
    PRETRAINED weights fail to load. At inference `pretrained=False`, the
    fallback cannot trigger, and the wrong variant is built.

    Nothing about that is loud. The class is the same, every other shape matches,
    and only two small tensors differ in name. So rather than guess, try both and
    keep the one whose state dict loads STRICTLY -- a non-strict load would
    "succeed" with a randomly initialised final norm, which is exactly the silent
    failure this function exists to prevent.
    """
    # fine_attn CANNOT be discovered by trying both: it adds no parameters, so
    # a fine_attn checkpoint loads strictly into a fine_attn=False model and
    # computes something else. It must be passed in from the run's config.json.
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    # head_type DOES change the parameter set, so a mismatch cannot load
    # strictly -- but the failure would surface as an exhausted fc_norm loop
    # with a confusing message. Name it here instead.
    # Sniffed by PRESENCE, and every branch must be positively identified. The
    # first version tested only for `mil_` and fell through to "hier", so a
    # `mean` checkpoint -- which has neither mil_ nor the hier stacks -- was
    # reported as 'hier' and refused with the arms named backwards. An
    # else-branch is a guess; these are three disjoint parameter sets, so ask.
    if any(k.startswith("mil_") for k in sd):
        ckpt_head = "mil"
    elif any(k.startswith(("slice_tf.", "study_tf.", "label_attn.")) for k in sd):
        ckpt_head = "hier"
    else:
        ckpt_head = "mean"      # no window-mixing parameters at all
    if ckpt_head != head_type:
        raise SystemExit(
            f"checkpoint is a '{ckpt_head}' head but head_type='{head_type}' was "
            f"requested -- read it from the run's config.json, as with fine_attn")
    # region_emb DOES add parameters, so a mismatch cannot load strictly -- but
    # it would surface as an exhausted fc_norm loop with a misleading message.
    # Name it here, as head_type does.
    ckpt_reg = sd["region_emb.weight"].shape[0] if "region_emb.weight" in sd else 0
    if ckpt_reg != n_regions:
        raise SystemExit(
            f"checkpoint has n_regions={ckpt_reg} but n_regions={n_regions} was "
            f"requested -- read it from the run's config.json, as with fine_attn")
    last = None
    for fc in (None, False, True):
        try:
            m = KneeModel(backbone, pretrained=False, img_size=img_size,
                          fc_norm=fc, fine_attn=fine_attn, head_type=head_type,
                          n_regions=n_regions, curia_config=curia_config)
            m.load_state_dict(sd, strict=True)          # strict: no silent gaps
            return m.to(device).eval()
        except (RuntimeError, TypeError) as e:
            last = e
    raise RuntimeError(f"no KneeModel variant accepts {ckpt}: {last}")
