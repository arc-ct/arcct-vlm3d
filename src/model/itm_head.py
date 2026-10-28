"""Image-Text Matching (ITM) head for Path 2.

A two-tower fusion head that takes the visual latent (pooled Q-Former output)
and the text [CLS] embedding, fuses them via concat + Linear + GELU, and
predicts a binary match score. Used with in-batch hard-negative mining: for
every (V_i, T_i) pair we draw one positive and one hardest negative.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ITMHead(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.fusion = nn.Linear(2 * dim, dim)
        self.classifier = nn.Linear(dim, 2)
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, visual: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """Predict match logits.

        Args
        ----
        visual : (N, D) pooled visual latents (one row per pair).
        text : (N, D) text [CLS] latents (one row per pair).

        Returns
        -------
        logits : (N, 2) match/no-match logits.
        """
        x = torch.cat([visual, text], dim=-1)
        x = F.gelu(self.fusion(x))
        return self.classifier(x)
