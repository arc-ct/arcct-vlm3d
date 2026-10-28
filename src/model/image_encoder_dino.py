"""DINO (v2/v3) 2D ViT as a 3D CT encoder, drop-in for RACImageEncoder.

Rationale: a published CT-RATE benchmark puts a FROZEN DINOv3-B linear probe at
0.798 macro AUC, well above CT-CLIP's 0.731 and above the 0.7546 that our own
Kinetics-only R3D-18 reaches un-adapted. Our two-stage pipeline lifted that
R3D-18 from 0.7546 to 0.8562, so the question this encoder exists to answer is
whether the same adaptation applied on top of a much stronger starting point
clears the deployed 0.8583.

DINO is a 2D model. Each axial slice is encoded independently and the per-slice
token grids are stacked into a volume, then pooled on the depth axis to the
12x12x12 grid the rest of the pipeline expects. Inter-slice context is therefore
supplied downstream (Q-Former) rather than by the backbone -- the same
slice-wise strategy the published 0.798 used.

Interface contract (must match model/image_encoder.RACImageEncoder exactly):
  FEAT_DIM = 512, IN_CHANNELS = 3
  forward_spatial(x[B,3,D,H,W]) -> [B, 512, 12, 12, 12]

Env:
  RAC_DINO_ID        HF id (default facebook/dinov3-vitb16-pretrain-lvd1689m;
                     DINOv3 repos are GATED -- needs an HF token. Set to
                     facebook/dinov2-base for an ungated fallback.)
  RAC_DINO_UNFREEZE  trailing blocks to train (default 4; 0 = fully frozen)
  RAC_DINO_SLICE_BS  slices per backbone chunk (default 96; lower if OOM)
  RAC_DINO_CKPT_GRAD gradient checkpointing (default 1)

Requires transformers >= 4.56 for DINOv3 (the `racv` env), >= 4.30 for DINOv2.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.image_encoder import RACImageEncoder

# DINO was pretrained on ImageNet-normalised RGB; our inputs are raw [0,1] HU
# windows. A mostly-frozen backbone cannot adapt to a shifted input
# distribution, so normalise here (the V-JEPA2 probe lost 0.005 external by
# skipping this).
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class DinoImageEncoder(nn.Module):
    """2D DINO ViT applied slice-wise, exposing an R3D-18-shaped feature grid."""

    FEAT_DIM = 512
    IN_CHANNELS = 3
    OUT_DEPTH = 12
    OUT_HW = 12

    global_pool = staticmethod(RACImageEncoder.global_pool)
    pool_organ_masks = staticmethod(RACImageEncoder.pool_organ_masks)
    pool_organ_masks_topk = staticmethod(RACImageEncoder.pool_organ_masks_topk)

    def __init__(self, pretrained_kinetics: bool = True):
        super().__init__()
        from transformers import AutoModel  # lazy: keeps `rac` env importable

        mid = os.environ.get("RAC_DINO_ID", "facebook/dinov3-vitb16-pretrain-lvd1689m")
        self.backbone = AutoModel.from_pretrained(mid)
        cfg = self.backbone.config
        self.hidden = int(cfg.hidden_size)
        self.patch = int(cfg.patch_size)
        # Slice side must be a whole number of patches; 192 works for patch 16,
        # 196 for patch 14. Resize rather than crop so no anatomy is lost.
        self.side = int(round(192 / self.patch)) * self.patch
        self.slice_bs = int(os.environ.get("RAC_DINO_SLICE_BS", "96"))

        self.project = nn.Conv3d(self.hidden, self.FEAT_DIM, kernel_size=1)
        self.register_buffer("_mean", torch.tensor(_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(_STD).view(1, 3, 1, 1), persistent=False)

        if os.environ.get("RAC_DINO_CKPT_GRAD", "1") == "1":
            try:
                self.backbone.gradient_checkpointing_enable()
            except Exception:
                pass
        self._freeze_except_last(int(os.environ.get("RAC_DINO_UNFREEZE", "4")))
        print(f"[DINO] {mid} hidden={self.hidden} patch={self.patch} side={self.side}", flush=True)

    def _blocks(self):
        for path in ("encoder.layer", "layer", "blocks", "encoder.layers"):
            obj = self.backbone
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return list(obj)
            except AttributeError:
                continue
        raise RuntimeError("could not locate transformer blocks on the DINO backbone")

    def _freeze_except_last(self, n: int) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if n <= 0:
            print("[DINO] backbone fully frozen", flush=True)
            return
        blocks = self._blocks()
        for blk in blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        tr = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.backbone.parameters())
        print(f"[DINO] unfroze last {n}/{len(blocks)} blocks -> {tr/1e6:.1f}M of {tot/1e6:.1f}M",
              flush=True)

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """[B,3,D,H,W] -> [B,512,12,12,12], matching the R3D-18 layer4 contract."""
        if x.ndim != 5 or x.shape[1] != self.IN_CHANNELS:
            raise ValueError(f"Expected [B,3,D,H,W], got {tuple(x.shape)}")
        B, _, D, H, W = x.shape

        # (B,3,D,H,W) -> (B*D,3,side,side): every axial slice is its own image.
        sl = x.permute(0, 2, 1, 3, 4).reshape(B * D, 3, H, W)
        if (H, W) != (self.side, self.side):
            sl = F.interpolate(sl, size=(self.side, self.side), mode="bilinear",
                               align_corners=False)
        sl = (sl - self._mean) / self._std

        g = self.side // self.patch
        outs = []
        for i in range(0, sl.shape[0], self.slice_bs):
            h = self.backbone(pixel_values=sl[i:i + self.slice_bs]).last_hidden_state
            # Drop CLS and any register tokens: patch tokens are the trailing g*g.
            outs.append(h[:, -g * g:, :])
        h = torch.cat(outs, 0)                                   # (B*D, g*g, hidden)

        grid = h.view(B, D, g, g, self.hidden).permute(0, 4, 1, 2, 3)   # (B,hid,D,g,g)
        grid = F.adaptive_avg_pool3d(grid, (self.OUT_DEPTH, self.OUT_HW, self.OUT_HW))
        return self.project(grid.to(self.project.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.global_pool(self.forward_spatial(x))
