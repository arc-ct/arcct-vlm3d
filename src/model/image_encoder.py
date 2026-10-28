"""3-channel R3D-18 image encoder for the model.

The final TIA recipe feeds clinical HU windows as three channels and keeps the
native torchvision/Kinetics R3D-18 stem. Spatial features are exposed directly
so global and all organ embeddings can be pooled from a single forward pass.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as models


class RACImageEncoder(nn.Module):
    """R3D-18 backbone with reusable layer4 spatial features."""

    FEAT_DIM = 512
    IN_CHANNELS = 3

    def __init__(self, pretrained_kinetics: bool = True):
        super().__init__()
        # RAC_R3D_ARCH picks the torchvision video backbone. All three share the
        # VideoResNet layout (stem/layer1-4) and a 512-channel layer4, so
        # forward_spatial and every downstream pooling path are unchanged. The
        # non-default arches exist to add architectural diversity to the
        # Stage-1 ensemble, where decorrelated errors are worth more than any
        # single model's accuracy.
        arch = os.environ.get("RAC_R3D_ARCH", "r3d_18").lower()
        _ARCHES = {
            "r3d_18": (models.r3d_18, "R3D_18_Weights"),
            "mc3_18": (models.mc3_18, "MC3_18_Weights"),
            "r2plus1d_18": (models.r2plus1d_18, "R2Plus1D_18_Weights"),
        }
        if arch not in _ARCHES:
            raise ValueError(f"RAC_R3D_ARCH={arch!r} not one of {sorted(_ARCHES)}")
        ctor, weights_attr = _ARCHES[arch]
        try:
            weights = getattr(models, weights_attr).KINETICS400_V1 if pretrained_kinetics else None
            backbone = ctor(weights=weights)
        except AttributeError:
            backbone = ctor(pretrained=pretrained_kinetics)
        except Exception as exc:
            if pretrained_kinetics and os.environ.get("RAC_KINETICS_REQUIRED", "1") == "1":
                raise RuntimeError(
                    f"Kinetics {arch} weights were requested but could not be loaded. "
                    "Set TORCH_HOME to a writable warmed cache, or set RAC_KINETICS_REQUIRED=0 "
                    "only for smoke/debug runs."
                ) from exc
            print(f"[RACImageEncoder] Kinetics weight load failed ({exc}); using random {arch} init")
            backbone = ctor(weights=None)
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Variant B: multi-scale feature pyramid (layer3 24^3 + layer4 12^3
        # trilinearly upsampled to 24^3, concat -> 768 channels, 1x1x1 project
        # back to 512 so to_visual_latent still consumes 512-d). Env-gated;
        # default off keeps legacy behaviour.
        self.use_multiscale = os.environ.get("RAC_USE_MULTISCALE", "0") == "1"
        if self.use_multiscale:
            self.ms_project = nn.Conv3d(256 + 512, self.FEAT_DIM, kernel_size=1, bias=False)
            # Zero-init the layer3 (first 256) input channels so the projection
            # starts as a pure layer4 pass-through. The first updates then
            # gradually let layer3 features contribute, instead of dumping a
            # large random perturbation onto a warm-started stage1 encoder.
            with torch.no_grad():
                w = self.ms_project.weight                          # [512, 768, 1, 1, 1]
                w.zero_()
                # Identity from layer4 channels to output channels.
                for c in range(self.FEAT_DIM):
                    w[c, 256 + c, 0, 0, 0] = 1.0
        else:
            self.ms_project = None

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Return layer4 map [B, 512, D', H', W'] without flattening.

        When ``RAC_USE_MULTISCALE=1`` returns the concatenated layer3+layer4
        feature pyramid projected back to 512 channels at the layer3 spatial
        resolution (24^3 for a 96x192x192 input).
        """
        if x.ndim != 5 or x.shape[1] != self.IN_CHANNELS:
            raise ValueError(f"Expected [B,3,D,H,W], got {tuple(x.shape)}")
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        if self.use_multiscale and self.ms_project is not None:
            l3 = x                                                       # [B, 256, 24, 24, 24]
            l4 = self.backbone.layer4(l3)                                # [B, 512, 12, 12, 12]
            l4_up = F.interpolate(l4, size=l3.shape[-3:], mode="trilinear", align_corners=False)
            cat = torch.cat([l3, l4_up], dim=1)                          # [B, 768, 24, 24, 24]
            return self.ms_project(cat)                                  # [B, 512, 24, 24, 24]
        x = self.backbone.layer4(x)
        return x

    @staticmethod
    def global_pool(feat_map: torch.Tensor) -> torch.Tensor:
        return feat_map.mean(dim=(2, 3, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.global_pool(self.forward_spatial(x))

    @staticmethod
    def pool_organ_masks(
        feat_map: torch.Tensor,
        masks_fine: torch.Tensor,
        organ_labels: list[int] | tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool all requested organs from one feature map.

        Args:
            feat_map: [B, C, Dp, Hp, Wp]
            masks_fine: [B, 1, D, H, W] or [B, D, H, W], integer labels
            organ_labels: TotalSegmentator label ids to pool

        Returns:
            roi: [B, O, C]
            valid: [B, O] true when the organ has at least one input voxel
        """
        if masks_fine.ndim == 5:
            masks = masks_fine[:, 0]
        elif masks_fine.ndim == 4:
            masks = masks_fine
        else:
            raise ValueError(f"Expected mask rank 4 or 5, got {masks_fine.ndim}")

        B, C, Dp, Hp, Wp = feat_map.shape
        labels = torch.as_tensor(organ_labels, device=masks.device, dtype=masks.dtype)
        organ_masks = masks.unsqueeze(1).eq(labels.view(1, -1, 1, 1, 1)).float()
        O = organ_masks.shape[1]
        valid = organ_masks.flatten(2).sum(dim=2) > 0

        mask_ds = F.adaptive_max_pool3d(organ_masks.reshape(B * O, 1, *masks.shape[-3:]), (Dp, Hp, Wp))
        mask_ds = mask_ds.reshape(B, O, Dp, Hp, Wp)
        denom = mask_ds.flatten(2).sum(dim=2).clamp(min=1.0)
        roi = (feat_map.unsqueeze(1) * mask_ds.unsqueeze(2)).flatten(3).sum(dim=3) / denom.unsqueeze(-1)

        global_roi = feat_map.mean(dim=(2, 3, 4)).unsqueeze(1).expand(B, O, C)
        roi = torch.where(valid.unsqueeze(-1), roi, global_roi)
        return roi, valid

    @staticmethod
    def pool_organ_masks_topk(
        feat_map: torch.Tensor,
        masks_fine: torch.Tensor,
        organ_labels: list[int] | tuple[int, ...],
        k: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Variant-A pooling: average the k highest-norm voxels inside each organ.

        Picks per-voxel L2 norm of the feature map, selects the top-k voxels
        within each organ mask, returns their mean. Falls back to global mean
        when an organ is absent. Same return signature as ``pool_organ_masks``.

        Args:
            feat_map: [B, C, Dp, Hp, Wp]
            masks_fine: [B, 1, D, H, W] or [B, D, H, W]
            organ_labels: TotalSegmentator organ ids
            k: number of top-norm voxels to keep per organ
        """
        if masks_fine.ndim == 5:
            masks = masks_fine[:, 0]
        elif masks_fine.ndim == 4:
            masks = masks_fine
        else:
            raise ValueError(f"Expected mask rank 4 or 5, got {masks_fine.ndim}")

        B, C, Dp, Hp, Wp = feat_map.shape
        labels_t = torch.as_tensor(organ_labels, device=masks.device, dtype=masks.dtype)
        organ_masks = masks.unsqueeze(1).eq(labels_t.view(1, -1, 1, 1, 1)).float()
        O = organ_masks.shape[1]
        valid = organ_masks.flatten(2).sum(dim=2) > 0

        mask_ds = F.adaptive_max_pool3d(
            organ_masks.reshape(B * O, 1, *masks.shape[-3:]), (Dp, Hp, Wp)
        ).reshape(B, O, Dp, Hp, Wp)

        N = Dp * Hp * Wp
        # Per-voxel L2 norm of the feature map; broadcast over organs.
        feat_norm = feat_map.norm(dim=1)                       # [B, Dp, Hp, Wp]
        feat_norm_flat = feat_norm.reshape(B, 1, N).expand(B, O, N)
        mask_flat = mask_ds.reshape(B, O, N)
        # Penalize voxels outside the organ so topk picks inside the mask.
        neg_inf = torch.finfo(feat_norm_flat.dtype).min
        scores = torch.where(mask_flat > 0.5, feat_norm_flat, torch.full_like(feat_norm_flat, neg_inf))
        # Pick top-k voxel indices per (B, O). Clamp k by available voxels so a
        # small organ that only has, say, 3 voxels picks 3 instead of crashing.
        voxel_counts = mask_flat.sum(dim=2).long().clamp(min=1)
        k_eff = int(min(k, N))
        topk_vals, topk_idx = scores.topk(k_eff, dim=2)        # [B, O, k]
        # Build a 0/1 weight per (B, O, N) for the picked voxels; zero out picks
        # that landed on an invalid (-inf) score because the organ had < k voxels.
        feat_flat = feat_map.reshape(B, C, N)                  # [B, C, N]
        # Gather: for each [B, O, k] -> feature [B, O, k, C]
        gather_idx = topk_idx.unsqueeze(-1).expand(B, O, k_eff, C)  # [B, O, k, C]
        feat_perm = feat_flat.permute(0, 2, 1).unsqueeze(1).expand(B, O, N, C)
        picked = feat_perm.gather(2, gather_idx)               # [B, O, k, C]
        # Mask out picks that came from -inf (those have no real organ voxel).
        valid_pick = topk_vals > neg_inf / 2
        valid_pick_f = valid_pick.float().unsqueeze(-1)        # [B, O, k, 1]
        denom = valid_pick_f.sum(dim=2).clamp(min=1.0)         # [B, O, 1]
        roi = (picked * valid_pick_f).sum(dim=2) / denom       # [B, O, C]

        global_roi = feat_map.mean(dim=(2, 3, 4)).unsqueeze(1).expand(B, O, C)
        roi = torch.where(valid.unsqueeze(-1), roi, global_roi)
        return roi, valid

    def forward_organ(self, x: torch.Tensor, organ_mask: torch.Tensor) -> torch.Tensor:
        feat_map = self.forward_spatial(x)
        if organ_mask.ndim == 4:
            organ_mask = organ_mask.unsqueeze(1)
        mask_ds = F.adaptive_max_pool3d(organ_mask.float(), feat_map.shape[-3:]).squeeze(1)
        denom = mask_ds.flatten(1).sum(dim=1).clamp(min=1.0)
        roi = (feat_map * mask_ds.unsqueeze(1)).flatten(2).sum(dim=2) / denom.unsqueeze(1)
        empty = organ_mask.flatten(1).sum(dim=1) == 0
        if empty.any():
            roi[empty] = self.global_pool(feat_map)[empty]
        return roi


def _compatible_state_dict(module: nn.Module, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    current = module.state_dict()
    out = {}
    for key, value in state.items():
        key2 = key
        if key2.startswith("backbone."):
            key2 = key2[len("backbone."):]
        if key2 in current and current[key2].shape == value.shape:
            out[key2] = value
    return out


def load_from_stage1(encoder: RACImageEncoder, ckpt_path: str) -> RACImageEncoder:
    """Load compatible stage-1 image encoder weights, skipping mismatched heads/stems."""
    pkg = torch.load(ckpt_path, map_location="cpu")
    state = pkg.get("image_encoder", pkg.get("model", pkg))
    bb = _compatible_state_dict(encoder.backbone, state)
    missing, unexpected = encoder.backbone.load_state_dict(bb, strict=False)
    print(f"[Stage1->RAC] Loaded compatible weights from {ckpt_path}")
    print(f"  loaded={len(bb)} missing={len(missing)} unexpected={len(unexpected)}")

    # Restore encoder-level params that live OUTSIDE .backbone -- e.g. the
    # 768->512 `project` conv on the ViT wrappers (DINO, V-JEPA2), which stage-1
    # trains. R3D-18 has none, so this is a no-op there. Without it the
    # projection is silently re-initialised and stage-1's adaptation of it is
    # thrown away.
    own = encoder.state_dict()
    extra = {k: v for k, v in state.items()
             if not k.startswith("backbone.") and k in own and own[k].shape == v.shape}
    if extra:
        encoder.load_state_dict(extra, strict=False)
        print(f"  restored {len(extra)} non-backbone tensor(s): {sorted(extra)[:4]}")
    return encoder
