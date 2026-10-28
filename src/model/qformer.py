"""BLIP-2-style Q-Former bridge for the model (Path 1).

Replaces the global-pool + ``to_visual_latent`` projection with K=32 learnable
query tokens that cross-attend to R3D-18 ``layer4`` spatial features. The 32
query tokens are pooled (mean over tokens by default; optionally use the first
"CLS" query) into a single ``dim``-d latent that feeds the existing CLIP
InfoNCE loss exactly like ``to_visual_latent`` output.

The module also exposes the pre-pool query tokens so downstream Image-Text
Matching (Path 2) and a generative decoder (Path 4) can consume them.

Q-Former takes the FLATTENED spatial map ``(B, N, image_dim)`` where ``N``
is ``D * H * W`` of the layer4 / multi-scale feature map. The forward
accepts either a 5D feature map or a pre-flattened token sequence and a
``key_padding_mask`` for Path 3 (anatomy-bound) variant.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QFormerBlock(nn.Module):
    """Self-attn + cross-attn + FFN block (BLIP-2 style)."""

    def __init__(self, dim: int = 768, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(
        self,
        q: torch.Tensor,
        img_tokens: torch.Tensor,
        cross_attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ):
        """One block forward.

        Args
        ----
        q : (B, Q, D) query tokens.
        img_tokens : (B, N, D) image tokens.
        cross_attn_mask : optional ``(B*num_heads, Q, N)`` additive mask for
            cross-attention. Used by Path 3 (anatomy-bound) to restrict each
            query's attendable voxels.
        key_padding_mask : optional ``(B, N)`` boolean mask where True means
            "ignore this key". Not used in normal flow.
        return_attn : when True, also return the per-head cross-attn weights
            (B, num_heads, Q, N). Used for attention-supervision loss.
        """
        h = self.norm1(q)
        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        q = q + attn_out

        h = self.norm2(q)
        cross_out, attn_w = self.cross_attn(
            h, img_tokens, img_tokens,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=False if return_attn else True,
        )
        q = q + cross_out

        q = q + self.ffn(self.norm3(q))
        if return_attn:
            return q, attn_w
        return q


class QFormer(nn.Module):
    """Learnable-query cross-attention bridge.

    Args
    ----
    num_queries : K query tokens (default 32).
    dim : query/latent dim (default 768 to match CLIP text latent).
    depth : number of QFormerBlocks.
    num_heads : MHA heads.
    image_dim : channel dim of the input feature map (R3D-18 layer4 = 512).
    pool : "mean" averages over the 32 queries; "cls" uses query 0.
    """

    def __init__(
        self,
        num_queries: int = 32,
        dim: int = 768,
        depth: int = 4,
        num_heads: int = 8,
        image_dim: int = 512,
        dropout: float = 0.0,
        pool: str = "mean",
        num_global_pool: int | None = None,
    ):
        super().__init__()
        self.num_queries = int(num_queries)
        self.dim = int(dim)
        self.image_dim = int(image_dim)
        if pool not in ("mean", "cls", "global"):
            raise ValueError(f"unknown pool={pool}")
        self.pool = pool
        # For pool="global": average only the last `num_global_pool` query tokens
        # (the unrestricted global queries appended by AnatomyQFormer). Defaults to
        # all queries, so pool="global" with no override equals "mean".
        self.num_global_pool = int(num_global_pool) if num_global_pool else int(num_queries)

        # Learnable query tokens.
        self.queries = nn.Parameter(torch.zeros(num_queries, dim))
        nn.init.normal_(self.queries, std=0.02)

        # Project image features to query dim.
        self.image_proj = nn.Linear(image_dim, dim)
        self.image_norm = nn.LayerNorm(dim)

        # Stack of blocks.
        self.layers = nn.ModuleList(
            [QFormerBlock(dim=dim, num_heads=num_heads, dropout=dropout) for _ in range(depth)]
        )
        self.norm_out = nn.LayerNorm(dim)

        # Xavier init for the linear layers inside blocks (queries already normal).
        self._reset_linear()

    def _reset_linear(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.image_proj:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # The image projection starts as a small random matrix so the queries
        # don't see a saturated feature map on update 1.
        nn.init.xavier_uniform_(self.image_proj.weight)
        nn.init.zeros_(self.image_proj.bias)

    @staticmethod
    def _flatten_feat_map(feat_map: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Convert (B, C, D, H, W) -> (B, N, C) and return the spatial shape."""
        if feat_map.ndim != 5:
            raise ValueError(f"expected 5D feature map (B,C,D,H,W), got {tuple(feat_map.shape)}")
        B, C, D, H, W = feat_map.shape
        return feat_map.flatten(2).transpose(1, 2).contiguous(), (D, H, W)

    def forward(
        self,
        feat_map: torch.Tensor,
        cross_attn_mask: torch.Tensor | None = None,
        return_tokens: bool = False,
        return_attn: bool = False,
    ):
        """Run the Q-Former.

        Args
        ----
        feat_map : (B, C, D, H, W) layer4 / multi-scale feature map. Channel
            dim must equal ``image_dim``.
        cross_attn_mask : optional ``(B*num_heads, Q, N)`` additive float mask
            for cross-attention. Path 3 supplies a role-mask via
            :class:`AnatomyQFormer`.
        return_tokens : when True, return ``(pooled, tokens)`` where ``tokens``
            is the pre-pool ``(B, Q, D)`` sequence (used by ITM head and
            generative decoder).

        Returns
        -------
        Tensor ``(B, D)`` pooled latent OR tuple ``(pooled, tokens)``.
        """
        img_tokens, _ = self._flatten_feat_map(feat_map)
        img_tokens = self.image_proj(img_tokens)
        img_tokens = self.image_norm(img_tokens)

        B = img_tokens.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1).contiguous()
        last_attn = None
        n_layers = len(self.layers)
        for i, layer in enumerate(self.layers):
            is_last = (i == n_layers - 1)
            if return_attn and is_last:
                q, last_attn = layer(q, img_tokens, cross_attn_mask=cross_attn_mask, return_attn=True)
            else:
                q = layer(q, img_tokens, cross_attn_mask=cross_attn_mask)
        q = self.norm_out(q)

        if self.pool == "mean":
            pooled = q.mean(dim=1)
        elif self.pool == "global":
            pooled = q[:, -self.num_global_pool:].mean(dim=1)
        else:  # "cls"
            pooled = q[:, 0]

        if return_attn and return_tokens:
            return pooled, q, last_attn
        if return_attn:
            return pooled, last_attn
        if return_tokens:
            return pooled, q
        return pooled
