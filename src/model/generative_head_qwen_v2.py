"""Stage-2 generative report head, v2 (VLM3D Task-1).

Additive rewrite of ``generative_head_qwen.py``.  The v1 head is imported by the
shipped Task-1 container (``VLM3D/reportgen_docker/process_reportgen.py``) and is
left untouched; everything new lives here.

Fixes over v1
-------------
1. **Decoding.**  v1 called ``llm.generate()`` with no decoding arguments, so it
   inherited ``Qwen2.5-3B-Instruct``'s ``generation_config.json``
   (``do_sample=true, temperature=0.7, top_p=0.8, top_k=20,
   repetition_penalty=1.05``).  Sampling is strictly dominated on a corpus-BLEU
   task.  v2 builds an explicit ``GenerationConfig`` (deterministic beam search,
   no sampling params at all) so nothing is inherited.
2. **Target length.**  CT-RATE reports are 330 Qwen tokens on average (p90 513);
   v1's 160-token budget truncated 96% of training targets, which is what drove
   the brevity penalty (BP=0.325) that lost the BLEU comparison.  v2 defaults to
   512 and pads to the longest sequence in the batch rather than to
   ``max_text_len`` (same math, ~40% less compute).
3. **EOS.**  Qwen's ``pad_token`` is ``<|endoftext|>`` (151643) and is *not*
   None, so v1's ``if pad_token is None: pad_token = eos_token`` branch never
   ran; the tokenizer never appends EOS to plain text, so ``<|im_end|>`` was
   never a loss target and the model could not learn to stop.  v2 appends EOS
   (151645, distinct from the pad id, so it survives the loss mask) to every
   target.
4. **First token.**  v1 scored ``logits[:, K:-1]`` against ``input_ids[:, 1:]``,
   dropping the ``last-visual-token -> first-report-token`` transition -- the
   exact transition ``generate()`` has to make from a pure visual prefix.  v2
   scores from ``K-1`` so that transition is supervised.

Conditioning
------------
``ClinicalInformation_EN`` exists in the reports CSV but is *not* available at
challenge inference time (the container is handed a bare ``.mha``), so it cannot
be used.  What *is* available at inference is our own abnormality classifier --
the strongest part of this system.  ``labels_to_prompt`` turns a label vector
into a short text prefix that is fed (loss-masked) between the visual tokens and
the report.  Train-time vectors are corrupted at roughly the classifier's
measured error rate so the model does not learn to over-trust a clean prompt it
will never see at test time.
"""

from __future__ import annotations

import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

DEFAULT_QWEN = "Qwen/Qwen2.5-3B-Instruct"

PATHOLOGY_NAMES = [
    "medical material", "arterial wall calcification", "cardiomegaly",
    "pericardial effusion", "coronary artery wall calcification", "hiatal hernia",
    "lymphadenopathy", "emphysema", "atelectasis", "lung nodule", "lung opacity",
    "pulmonary fibrotic sequela", "pleural effusion", "mosaic attenuation pattern",
    "peribronchial thickening", "consolidation", "bronchiectasis",
    "interlobular septal thickening",
]

PROMPT_TEMPLATE = "Chest CT. Detected: {findings}.\nReport:"
PROMPT_NEGATIVE = "no acute abnormality"


def labels_to_prompt(label_vec, names: list[str] | None = None,
                     fn_rate: float = 0.0, fp_rate: float = 0.0,
                     rng: random.Random | None = None) -> str:
    """Render a binary abnormality vector as a short conditioning prefix.

    ``fn_rate`` drops a true positive, ``fp_rate`` adds a false positive; both
    default to 0 (eval).  During training they simulate the classifier noise the
    model will actually be conditioned on at inference.
    """
    names = names or PATHOLOGY_NAMES
    rng = rng or random
    pos = []
    for i, v in enumerate(label_vec):
        on = float(v) >= 0.5
        if on and fn_rate > 0 and rng.random() < fn_rate:
            on = False
        elif (not on) and fp_rate > 0 and rng.random() < fp_rate:
            on = True
        if on and i < len(names):
            pos.append(names[i])
    return PROMPT_TEMPLATE.format(findings=", ".join(pos) if pos else PROMPT_NEGATIVE)


class QwenReportDecoder(nn.Module):
    """Q-Former tokens -> trainable linear bridge -> frozen Qwen2.5-3B.

    The checkpoint contract is unchanged: a single ``lm_bridge`` state dict for
    an ``nn.Linear(qformer_dim, llm_dim)``, which ``eval.py::build_model_for_eval``
    and the shipped container already know how to load.
    """

    def __init__(
        self,
        qformer_dim: int = 768,
        freeze_llm: bool = True,
        max_text_len: int = 512,
        max_prompt_len: int = 96,
        llm_name: str = DEFAULT_QWEN,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.pad_id = self.tokenizer.pad_token_id
        # Qwen: pad=<|endoftext|>(151643), eos=<|im_end|>(151645).  Distinct ids,
        # so an appended EOS is a real loss target and is not swallowed by the
        # pad mask.  If a tokenizer ever ties them, fall back to a stop id that
        # generate() can still use.
        self.eos_id = self.tokenizer.eos_token_id
        assert self.eos_id is not None, "tokenizer has no eos_token_id"

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.llm_dim = self.llm.config.hidden_size
        self.max_text_len = int(max_text_len)
        self.max_prompt_len = int(max_prompt_len)

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

    # ---------------------------------------------------------------- batching
    @torch.no_grad()
    def _build_batch(self, texts: list[str], prompts: list[str] | None,
                     device: torch.device):
        """[prompt][report][EOS] ids, right-padded, with a -100 label mask.

        Prompt positions are masked out of the loss: the prompt is conditioning,
        not something we want the bridge to learn to reproduce.
        """
        B = len(texts)
        tgt = self.tokenizer(texts, add_special_tokens=False,
                             truncation=True, max_length=self.max_text_len - 1)["input_ids"]
        if prompts is not None:
            pr = self.tokenizer(prompts, add_special_tokens=False,
                                truncation=True, max_length=self.max_prompt_len)["input_ids"]
        else:
            pr = [[] for _ in range(B)]

        seqs, labs = [], []
        for p, t in zip(pr, tgt):
            ids = list(p) + list(t) + [self.eos_id]
            lab = [-100] * len(p) + list(t) + [self.eos_id]
            seqs.append(ids)
            labs.append(lab)

        T = max(len(s) for s in seqs)
        input_ids = torch.full((B, T), self.pad_id, dtype=torch.long)
        labels = torch.full((B, T), -100, dtype=torch.long)
        attn = torch.zeros((B, T), dtype=torch.long)
        for i, (s, l) in enumerate(zip(seqs, labs)):
            n = len(s)
            input_ids[i, :n] = torch.tensor(s, dtype=torch.long)
            labels[i, :n] = torch.tensor(l, dtype=torch.long)
            attn[i, :n] = 1
        return input_ids.to(device), attn.to(device), labels.to(device)

    # ---------------------------------------------------------------- training
    def forward(self, visual_tokens: torch.Tensor, texts: list[str],
                prompts: list[str] | None = None) -> torch.Tensor:
        B, K, _ = visual_tokens.shape
        device = visual_tokens.device
        input_ids, attn_mask, labels = self._build_batch(texts, prompts, device)

        visual_embeds = self.bridge(visual_tokens.float()).to(self.llm.dtype)
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        full_mask = torch.cat(
            [torch.ones(B, K, dtype=attn_mask.dtype, device=device), attn_mask], dim=1
        )

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            use_cache=False,
        )
        # position K-1 (last visual token) predicts ids[:, 0] -- the transition
        # generate() actually has to make.  v1 dropped it.
        logits = outputs.logits[:, K - 1:-1, :]
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            labels.reshape(-1),
            ignore_index=-100,
        )

    # -------------------------------------------------------------- generation
    def generation_config(self, **overrides) -> GenerationConfig:
        """Deterministic beam search, built from scratch.

        Constructed fresh rather than merged into ``self.llm.generation_config``
        so none of Qwen's shipped sampling parameters can leak in.
        ``repetition_penalty`` stays at 1.0 on purpose: radiology boilerplate is
        legitimately repetitive and penalising it deletes n-grams that earn BLEU.
        """
        cfg = dict(
            num_beams=int(os.environ.get("RG_NUM_BEAMS", "4")),
            do_sample=False,
            early_stopping=True,
            min_new_tokens=int(os.environ.get("RG_MIN_NEW_TOKENS", "200")),
            max_new_tokens=int(os.environ.get("RG_MAX_NEW_TOKENS", "420")),
            length_penalty=float(os.environ.get("RG_LENGTH_PENALTY", "1.1")),
            repetition_penalty=float(os.environ.get("RG_REPETITION_PENALTY", "1.0")),
            no_repeat_ngram_size=int(os.environ.get("RG_NO_REPEAT_NGRAM", "0")),
            eos_token_id=self.eos_id,
            pad_token_id=self.pad_id,
        )
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        return GenerationConfig(**cfg)

    @torch.no_grad()
    def _prefix_embeds(self, visual_tokens: torch.Tensor,
                       prompts: list[str] | None):
        """[visual][prompt] embeds, LEFT-padded (decoder-only generation)."""
        B, K, _ = visual_tokens.shape
        device = visual_tokens.device
        visual_embeds = self.bridge(visual_tokens.float()).to(self.llm.dtype)
        if not prompts:
            attn = torch.ones(B, K, dtype=torch.long, device=device)
            return visual_embeds, attn

        pr = self.tokenizer(prompts, add_special_tokens=False,
                            truncation=True, max_length=self.max_prompt_len)["input_ids"]
        P = max(len(p) for p in pr)
        emb_layer = self.llm.get_input_embeddings()
        pad_emb = emb_layer(torch.tensor([self.pad_id], device=device))[0]
        outs, masks = [], []
        for i, p in enumerate(pr):
            n_pad = P - len(p)
            ids = torch.tensor(p, dtype=torch.long, device=device)
            p_emb = emb_layer(ids) if len(p) else torch.zeros(0, self.llm_dim,
                                                             device=device, dtype=visual_embeds.dtype)
            left = pad_emb.unsqueeze(0).expand(n_pad, -1) if n_pad else \
                torch.zeros(0, self.llm_dim, device=device, dtype=visual_embeds.dtype)
            outs.append(torch.cat([left, visual_embeds[i], p_emb], dim=0))
            masks.append(torch.cat([
                torch.zeros(n_pad, dtype=torch.long, device=device),
                torch.ones(K + len(p), dtype=torch.long, device=device),
            ]))
        return torch.stack(outs, 0), torch.stack(masks, 0)

    @torch.no_grad()
    def generate(self, visual_tokens: torch.Tensor,
                 prompts: list[str] | None = None, **overrides) -> list[str]:
        embeds, attn = self._prefix_embeds(visual_tokens, prompts)
        out = self.llm.generate(
            inputs_embeds=embeds,
            attention_mask=attn,
            generation_config=self.generation_config(**overrides),
        )
        return [t.strip() for t in self.tokenizer.batch_decode(out, skip_special_tokens=True)]
