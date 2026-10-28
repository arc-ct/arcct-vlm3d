"""Stage-2 BLIP-2 generative head with frozen Qwen-2.5-3B-Instruct.

Mirrors the GPT-2 interface in ``generative_head.py`` but uses a 3B Qwen
backbone. Larger LLM gives richer language modeling signal to flow back
through the bridge into the Q-Former.

Stage-2 paradigm:
- LLM is frozen (no grad)
- Bridge (linear projection) is trainable
- Q-Former weights may be loaded from Stage-1 ckpt and either frozen or
  finetuned via ``freeze_qformer`` controlled in train script.

VRAM budget (Qwen-2.5-3B fp16): ~6 GB + ~2 GB activations at bs=4, seq=160.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_QWEN = "Qwen/Qwen2.5-3B-Instruct"


class QwenGenerativeDecoder(nn.Module):
    def __init__(
        self,
        qformer_dim: int = 768,
        freeze_llm: bool = True,
        max_text_len: int = 160,
        llm_name: str = DEFAULT_QWEN,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.llm_dim = self.llm.config.hidden_size
        self.max_text_len = int(max_text_len)

        self.bridge = nn.Linear(qformer_dim, self.llm_dim)
        nn.init.xavier_uniform_(self.bridge.weight)
        nn.init.zeros_(self.bridge.bias)

        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False
            self.llm.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        for p in self.llm.parameters():
            if not p.requires_grad:
                self.llm.eval()
                break
        return self

    @torch.no_grad()
    def _tokenize(self, texts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
        )
        return enc["input_ids"].to(device), enc["attention_mask"].to(device)

    def forward(self, visual_tokens: torch.Tensor, texts: list[str]) -> torch.Tensor:
        B, K, D = visual_tokens.shape
        device = visual_tokens.device
        input_ids, attn_mask = self._tokenize(texts, device)
        T = input_ids.shape[1]

        visual_embeds = self.bridge(visual_tokens.float()).to(self.llm.dtype)
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        full_mask = torch.cat(
            [torch.ones(B, K, dtype=attn_mask.dtype, device=device), attn_mask],
            dim=1,
        )

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            use_cache=False,
        )
        logits = outputs.logits[:, K:-1, :].float()
        targets = input_ids[:, 1:]
        pad_id = self.tokenizer.pad_token_id
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=pad_id,
        )
        return loss

    @torch.no_grad()
    def generate(self, visual_tokens: torch.Tensor, max_new_tokens: int = 160, **gen_kwargs) -> list[str]:
        B, K, _ = visual_tokens.shape
        device = visual_tokens.device
        visual_embeds = self.bridge(visual_tokens.float()).to(self.llm.dtype)
        attn_mask = torch.ones(B, K, dtype=torch.long, device=device)
        out = self.llm.generate(
            inputs_embeds=visual_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            **gen_kwargs,
        )
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)
