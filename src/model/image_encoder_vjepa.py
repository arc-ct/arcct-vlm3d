"""V-JEPA2 ViT-L image encoder, drop-in for RACImageEncoder.

Why this exists: a frozen-feature probe (2026-08-03) showed V-JEPA2's raw
representation beats a Kinetics-only R3D-18 by +0.021 external macro-AUROC, but
that stage-1 supervised ADAPTATION is worth ~4x more (+0.078). A frozen V-JEPA2
therefore cannot compete; the encoder only earns its keep if it is adapted. Full
fine-tuning of a 326M ViT-L does not fit the compute budget, so the last N
transformer blocks are unfrozen and everything below stays frozen.

Interface contract (must match model/image_encoder.RACImageEncoder exactly):
  FEAT_DIM = 512, IN_CHANNELS = 3
  forward_spatial(x[B,3,D,H,W]) -> [B, 512, 12, 12, 12]

The 12x12x12 output grid is deliberate. V-JEPA2 emits (D/2, H/16, W/16) =
48x12x12 tokens for our 96x192x192 input; the depth axis is pooled 48->12 so the
grid is byte-compatible with the R3D-18 layer4 map. That keeps the Q-Former, the
organ-mask pooling and the deployed Slice Top-12 readout (which iterates
feat.shape[2] and expects 12 slabs) completely unchanged.

Env:
  RAC_VJEPA_ID        HF id                       (default facebook/vjepa2-vitl-fpc64-256)
  RAC_VJEPA_UNFREEZE  trailing blocks to train    (default 4; 0 = fully frozen)
  RAC_VJEPA_CKPT_GRAD gradient checkpointing      (default 1)

Requires transformers >= 4.52 (VJEPA2Model), i.e. the `racv` env -- NOT `rac`.
The import is lazy so that merely importing this module never breaks `rac`.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.image_encoder import RACImageEncoder

# V-JEPA2 was pretrained on ImageNet-normalised video. Our pipeline feeds raw
# [0,1] HU windows; stage-1 let R3D-18 adapt to that, but a mostly-frozen
# backbone cannot. Feeding the wrong distribution cost 0.005 external in the
# probe, so normalise here rather than relying on the caller.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class VJEPAImageEncoder(nn.Module):
    """V-JEPA2 ViT-L backbone exposing an R3D-18-shaped feature grid."""

    FEAT_DIM = 512
    IN_CHANNELS = 3
    OUT_DEPTH = 12

    # Reuse the pooling logic verbatim so organ ROIs are computed identically.
    global_pool = staticmethod(RACImageEncoder.global_pool)
    pool_organ_masks = staticmethod(RACImageEncoder.pool_organ_masks)
    pool_organ_masks_topk = staticmethod(RACImageEncoder.pool_organ_masks_topk)

    def __init__(self, pretrained_kinetics: bool = True):
        # `pretrained_kinetics` is accepted and ignored: it exists only so this
        # class is constructor-compatible with RACImageEncoder at the call sites
        # in train.py and pretrain_vision.py.
        super().__init__()
        from transformers import VJEPA2Model            # lazy: keeps `rac` importable

        model_id = os.environ.get("RAC_VJEPA_ID", "facebook/vjepa2-vitl-fpc64-256")
        self.backbone = VJEPA2Model.from_pretrained(model_id)
        self.hidden = self.backbone.config.hidden_size
        self.tubelet = self.backbone.config.tubelet_size
        self.patch = self.backbone.config.patch_size

        n_unfreeze = int(os.environ.get("RAC_VJEPA_UNFREEZE", "4"))
        self._freeze_except_last(n_unfreeze)

        if os.environ.get("RAC_VJEPA_CKPT_GRAD", "1") == "1" and n_unfreeze > 0:
            try:
                self.backbone.gradient_checkpointing_enable()
            except Exception as exc:                     # not fatal, only slower/heavier
                print(f"[VJEPA] gradient checkpointing unavailable ({exc})")

        # 1x1x1 projection back to the 512-d contract consumed by the Q-Former
        # and to_visual_latent.
        self.project = nn.Conv3d(self.hidden, self.FEAT_DIM, kernel_size=1, bias=False)

        self.register_buffer("_mean", torch.tensor(_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(_STD).view(1, 3, 1, 1, 1), persistent=False)

    # ── freezing ────────────────────────────────────────────────────────────
    def _blocks(self):
        enc = getattr(self.backbone, "encoder", self.backbone)
        for attr in ("layer", "layers", "blocks"):
            b = getattr(enc, attr, None)
            if b is not None:
                return b
        raise RuntimeError("could not locate V-JEPA2 transformer blocks")

    def _freeze_except_last(self, n: int):
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if n <= 0:
            print("[VJEPA] backbone fully frozen")
            return
        blocks = self._blocks()
        for blk in blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        tr = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"[VJEPA] unfroze last {n}/{len(blocks)} blocks -> {tr/1e6:.1f}M trainable "
              f"of {sum(p.numel() for p in self.backbone.parameters())/1e6:.1f}M")

    # ── forward ─────────────────────────────────────────────────────────────
    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """[B,3,D,H,W] -> [B,512,12,12,12], matching the R3D-18 layer4 contract."""
        if x.ndim != 5 or x.shape[1] != self.IN_CHANNELS:
            raise ValueError(f"Expected [B,3,D,H,W], got {tuple(x.shape)}")
        B, _, D, H, W = x.shape

        x = (x - self._mean) / self._std
        # V-JEPA2 takes [B, T, C, H, W] and treats T as time; depth-as-time is the
        # same convention the Kinetics R3D-18 stem already relies on.
        vid = x.permute(0, 2, 1, 3, 4).contiguous()
        h = self.backbone(pixel_values_videos=vid).last_hidden_state   # [B, N, hidden]

        d, hh, ww = D // self.tubelet, H // self.patch, W // self.patch
        if h.shape[1] != d * hh * ww:
            raise RuntimeError(f"token count {h.shape[1]} != {d}*{hh}*{ww}; "
                               f"input {D}x{H}x{W} not divisible by "
                               f"tubelet {self.tubelet}/patch {self.patch}")
        grid = h.view(B, d, hh, ww, self.hidden).permute(0, 4, 1, 2, 3)  # [B,hid,d,hh,ww]

        # Pool the depth axis down to 12 so downstream code (Q-Former, Slice
        # Top-12) sees exactly the grid it was built for.
        if d != self.OUT_DEPTH:
            grid = F.adaptive_avg_pool3d(grid, (self.OUT_DEPTH, hh, ww))
        return self.project(grid.to(self.project.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.global_pool(self.forward_spatial(x))


def build_image_encoder(pretrained_kinetics: bool = True):
    """Factory honouring RAC_ENCODER: 'r3d' (default) | 'vjepa'.

    Kept here rather than in image_encoder.py so that the V-JEPA2 path adds no
    import weight or failure mode to runs that do not ask for it.
    """
    kind = os.environ.get("RAC_ENCODER", "r3d").lower()
    if kind in ("r3d", "r3d18", ""):
        return RACImageEncoder(pretrained_kinetics=pretrained_kinetics)
    if kind in ("vjepa", "vjepa2"):
        return VJEPAImageEncoder(pretrained_kinetics=pretrained_kinetics)
    if kind in ("dino", "dinov2", "dinov3"):
        from model.image_encoder_dino import DinoImageEncoder
        return DinoImageEncoder(pretrained_kinetics=pretrained_kinetics)
    raise ValueError(f"unknown RAC_ENCODER={kind!r} (expected r3d|vjepa|dino)")
