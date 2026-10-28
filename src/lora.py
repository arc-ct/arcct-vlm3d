"""Small LoRA utilities for CXR-BERT text adaptation.

The final RAC recipe keeps the base CXR-BERT weights frozen and trains only
rank-r adapters on the last four query/value projections, plus the text
projection head in ``CTCLIP``.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap an ``nn.Linear`` with LoRA: base(x) + B(A(x)) * alpha / r."""

    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base_linear
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.zeros(r, base_linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B starts at zero so the wrapped module initially matches the base model.
        nn.init.zeros_(self.lora_B)
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A._is_lora_text = True
        self.lora_B._is_lora_text = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling


def apply_lora_to_bert(
    text_transformer: nn.Module,
    target_layer_indices=(8, 9, 10, 11),
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> int:
    """Apply LoRA to BERT self-attention query and value projections."""
    n_wrapped = 0
    layers = text_transformer.encoder.layer
    for idx in target_layer_indices:
        layer = layers[idx]
        attn = layer.attention.self
        if not isinstance(attn.query, LoRALinear):
            attn.query = LoRALinear(attn.query, r=r, alpha=alpha, dropout=dropout)
            n_wrapped += 1
        if not isinstance(attn.value, LoRALinear):
            attn.value = LoRALinear(attn.value, r=r, alpha=alpha, dropout=dropout)
            n_wrapped += 1
    return n_wrapped


def is_lora_parameter(param: nn.Parameter) -> bool:
    return bool(getattr(param, "_is_lora_text", False))
