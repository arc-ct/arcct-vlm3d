"""Evaluation for final the model.

Primary output is mask-free global prompt AUC. Anatomy-routed evaluation is also
computed when TotalSegmentator masks are available, but it is saved as a
secondary/deployable-TS analysis rather than the headline zero-shot metric.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dataset import (  # noqa: E402
    PATHOLOGIES, N_PATH, PATHOLOGY_FINE_ORGANS, RACDatasetV4, pathology_organ_lists, rac_collate,
)
from lora import apply_lora_to_bert  # noqa: E402
from model.image_encoder import RACImageEncoder  # noqa: E402
from ct_clip.ct_clip import CTCLIP  # noqa: E402
from model.qformer import QFormer  # noqa: E402
from model.anatomy_qformer import AnatomyQFormer  # noqa: E402
from model.itm_head import ITMHead  # noqa: E402


DATA_ROOT = os.environ.get("RAC_DATA_ROOT", "/data/data")
DATA_VALID = f"{DATA_ROOT}/mps_ct_npz/valid"
MASK_VALID = os.environ.get("RAC_MASK_VALID", "/data/data/fine_valid_masks_192")
REPORTS_VALID = f"{DATA_ROOT}/ct_reports/valid_reports.csv"
LABELS_VALID = os.environ.get("RAC_LABELS_VALID", f"{DATA_ROOT}/multi_abnormality_labels/valid_predicted_labels.csv")
CKPT = os.environ.get("EVAL_CKPT", "")
RESULTS_DIR = os.environ.get("EVAL_RESULTS_DIR", "")
BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "4"))
NUM_WORKERS = int(os.environ.get("EVAL_NUM_WORKERS", "4"))
N_BOOTSTRAP = int(os.environ.get("EVAL_BOOTSTRAP", "1000"))
THRESHOLD = float(os.environ.get("EVAL_THRESHOLD", "0.5"))
TEMPERATURE = float(os.environ.get("RAC_TEMPERATURE", "0.07"))
PROMPT_LEN = int(os.environ.get("RAC_PROMPT_LEN", "64"))
VAL_LIMIT = int(os.environ.get("RAC_VAL_LIMIT", "0"))
FAIL_FAST = os.environ.get("RAC_FAIL_FAST", "1") == "1"
ATEL_GLOBAL = os.environ.get("EVAL_ATEL_GLOBAL", "0") == "1"

# Q-Former family flags (mirror train.py). Default off so legacy CTCLIP ckpts
# load unchanged. When a ckpt was produced by a Q-Former training job the env
# vars don't need to be set: ``build_model_for_eval`` will auto-detect the
# ``qformer_module`` key and switch on the matching module.
USE_QFORMER = os.environ.get("RAC_USE_QFORMER", "0") == "1"
USE_ANATOMY_QFORMER = os.environ.get("RAC_USE_ANATOMY_QFORMER", "0") == "1"
USE_ITM = os.environ.get("RAC_USE_ITM", "0") == "1"
USE_GENERATIVE = os.environ.get("RAC_USE_GENERATIVE", "0") == "1"
QFORMER_NUM_QUERIES = int(os.environ.get("RAC_QFORMER_QUERIES", "32"))
QFORMER_DEPTH = int(os.environ.get("RAC_QFORMER_DEPTH", "4"))
QFORMER_HEADS = int(os.environ.get("RAC_QFORMER_HEADS", "8"))
QFORMER_POOL = os.environ.get("RAC_QFORMER_POOL", "mean")
QFORMER_DIM = 768

GLOBAL_EVAL_PATHOLOGIES = {"Mosaic attenuation pattern", "Medical material"}
if ATEL_GLOBAL:
    GLOBAL_EVAL_PATHOLOGIES.add("Atelectasis")

ORGAN_LABELS = sorted({o for organs in PATHOLOGY_FINE_ORGANS.values() for o in organs})
ORGAN_TO_INDEX = {organ: idx for idx, organ in enumerate(ORGAN_LABELS)}


def build_model_for_eval(ckpt_path):
    pkg = torch.load(ckpt_path, map_location="cpu")
    cfg = pkg.get("config", {})
    use_lora = bool(cfg.get("use_lora", True))
    lora_r = int(cfg.get("lora_r", os.environ.get("RAC_LORA_R", "8")))
    lora_alpha = int(cfg.get("lora_alpha", os.environ.get("RAC_LORA_ALPHA", "16")))

    tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
    text_encoder.resize_token_embeddings(len(tokenizer))
    if use_lora:
        n = apply_lora_to_bert(text_encoder, r=lora_r, alpha=lora_alpha)
        print(f"[eval] Applied LoRA structure to {n} projections for checkpoint load")

    # RAC_ENCODER must be honoured here too: train.py and pretrain_vision.py
    # dispatch on it, but eval.py used to hardcode the R3D-18. Loading a DINO or
    # V-JEPA2 checkpoint into an R3D-18 does not raise (strict=False) -- it just
    # leaves the encoder randomly initialised and scores ~chance.
    if os.environ.get("RAC_ENCODER", "r3d").lower() in ("r3d", "r3d18", ""):
        image_encoder = RACImageEncoder(pretrained_kinetics=False)
    else:
        from model.image_encoder_vjepa import build_image_encoder
        image_encoder = build_image_encoder(pretrained_kinetics=False)
    clip = CTCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        dim_image=512,
        dim_text=768,
        dim_latent=768,
        extra_latent_projection=False,
        use_mlm=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False,
    )
    missing, unexpected = clip.load_state_dict(pkg["model"], strict=False)
    print(f"[eval] Loaded {ckpt_path} best_auc={pkg.get('best_auc', '?')} missing={len(missing)} unexpected={len(unexpected)}")

    apply_ema = os.environ.get("RAC_EVAL_APPLY_EMA", "1") == "1"
    ema_state = pkg.get("ema")
    if apply_ema and ema_state:
        applied = 0
        for name, p in clip.named_parameters():
            if name in ema_state:
                p.data.copy_(ema_state[name].to(p.device, dtype=p.dtype))
                applied += 1
        print(f"[eval] Applied EMA shadow to clip: {applied}/{len(ema_state)} tensors matched")

    # Q-Former family auto-detect + load. The training script saves Q-Former /
    # ITM / generative-bridge state under fixed keys; presence of those keys
    # implies the matching module was active. ``RAC_USE_*`` env vars are
    # honored as an explicit override but auto-detect is preferred so the eval
    # wrapper does not need to re-declare flags.
    has_qf = "qformer_module" in pkg
    has_itm = "itm_module" in pkg
    has_lm = "lm_bridge" in pkg
    qf_state = pkg.get("qformer_module") if has_qf else None
    # Anatomy variant nests the inner QFormer under "qformer.*" keys.
    is_anatomy = bool(qf_state) and any(k.startswith("qformer.") for k in qf_state.keys())

    want_qf = USE_QFORMER or USE_ANATOMY_QFORMER or has_qf
    want_anatomy = USE_ANATOMY_QFORMER or is_anatomy
    want_itm = USE_ITM or has_itm
    want_lm = USE_GENERATIVE or has_lm

    clip.qformer_module = None
    clip.use_anatomy_qformer = False
    clip.itm_module = None
    clip.lm_bridge = None

    if want_qf:
        if has_qf and not (USE_QFORMER or USE_ANATOMY_QFORMER):
            kind = "Anatomy" if is_anatomy else "plain"
            print(f"[eval] auto-detected Q-Former in ckpt, enabling ({kind})")
        if want_anatomy:
            anatomy_labels = list(range(1, 11))
            pathology_labels = pathology_organ_lists(PATHOLOGIES)
            qformer_module = AnatomyQFormer(
                anatomy_labels=anatomy_labels,
                pathology_labels=pathology_labels,
                num_global=QFORMER_NUM_QUERIES - len(anatomy_labels) - len(pathology_labels),
                dim=QFORMER_DIM,
                depth=QFORMER_DEPTH,
                num_heads=QFORMER_HEADS,
                image_dim=RACImageEncoder.FEAT_DIM,
                pool=QFORMER_POOL,
            )
        else:
            qformer_module = QFormer(
                num_queries=QFORMER_NUM_QUERIES,
                dim=QFORMER_DIM,
                depth=QFORMER_DEPTH,
                num_heads=QFORMER_HEADS,
                image_dim=RACImageEncoder.FEAT_DIM,
                pool=QFORMER_POOL,
            )
        if qf_state is not None:
            m_qf, u_qf = qformer_module.load_state_dict(qf_state, strict=False)
            print(f"[eval] Loaded Q-Former state missing={len(m_qf)} unexpected={len(u_qf)}")
        else:
            print("[eval] WARNING Q-Former requested but ckpt has no qformer_module state")
        clip.qformer_module = qformer_module
        clip.use_anatomy_qformer = bool(want_anatomy)

    if want_itm:
        itm_module = ITMHead(dim=QFORMER_DIM)
        if has_itm:
            m_itm, u_itm = itm_module.load_state_dict(pkg["itm_module"], strict=False)
            print(f"[eval] Loaded ITM head missing={len(m_itm)} unexpected={len(u_itm)}")
        # ITM head is unused at eval (cosine-sim drives prompt AUC) but we
        # still construct + load it so deployers who want to use the head
        # offline get the trained weights.
        clip.itm_module = itm_module

    if want_lm:
        # train.py persists only the bridge linear (GPT-2 is frozen). Build a
        # bare nn.Linear to receive it; full GenerativeDecoder would force a
        # GPT-2 download that is pointless at eval.
        import torch.nn as nn
        out_dim = QFORMER_DIM
        if has_lm and "weight" in pkg["lm_bridge"]:
            out_dim = int(pkg["lm_bridge"]["weight"].shape[0])
        lm_bridge = nn.Linear(QFORMER_DIM, out_dim)
        if has_lm:
            m_lm, u_lm = lm_bridge.load_state_dict(pkg["lm_bridge"], strict=False)
            print(f"[eval] Loaded generative bridge dim={out_dim} missing={len(m_lm)} unexpected={len(u_lm)}")
        clip.lm_bridge = lm_bridge

    return clip, tokenizer


@torch.no_grad()
def encode_prompts(clip, tokenizer, device):
    pos = tokenizer([f"{p}." for p in PATHOLOGIES], return_tensors="pt", padding="max_length", truncation=True, max_length=PROMPT_LEN).to(device)
    neg = tokenizer([f"No {p}." for p in PATHOLOGIES], return_tensors="pt", padding="max_length", truncation=True, max_length=PROMPT_LEN).to(device)
    out_pos = clip.text_transformer(pos["input_ids"], pos["attention_mask"])
    out_neg = clip.text_transformer(neg["input_ids"], neg["attention_mask"])
    return (
        F.normalize(clip.to_text_latent(out_pos[0][:, 0, :]), dim=-1),
        F.normalize(clip.to_text_latent(out_neg[0][:, 0, :]), dim=-1),
    )


def prompt_probs(img_lat, pos_embs, neg_embs):
    img_lat = F.normalize(img_lat, dim=-1)
    sims_pos = img_lat @ pos_embs.T
    sims_neg = img_lat @ neg_embs.T
    stacked = torch.stack([sims_pos, sims_neg], dim=-1) / TEMPERATURE
    return F.softmax(stacked, dim=-1)[..., 0]


def _safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _bootstrap_indices(n, groups=None, rng=None):
    rng = np.random.default_rng(42) if rng is None else rng
    if groups is None:
        return rng.integers(0, n, size=n)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    sampled_groups = rng.choice(uniq, size=len(uniq), replace=True)
    idxs = [np.where(groups == g)[0] for g in sampled_groups]
    return np.concatenate(idxs)


def compute_metric_bundle(pred, true, class_names, accessions=None, threshold=0.5, n_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_hat = (pred >= threshold).astype(int)
    per_class = {}
    for j, name in enumerate(class_names):
        yt, ys, yh = true[:, j].astype(int), pred[:, j], y_hat[:, j]
        if len(np.unique(yt)) > 1:
            tn, fp, fn, tp = confusion_matrix(yt, yh, labels=[0, 1]).ravel()
            spec = float(tn / max(tn + fp, 1))
        else:
            spec = float("nan")
        per_class[name] = {
            "auc": _safe_auc(yt, ys),
            "acc": float(accuracy_score(yt, yh)),
            "f1": float(f1_score(yt, yh, zero_division=0)),
            "precision": float(precision_score(yt, yh, zero_division=0)),
            "sensitivity": float(recall_score(yt, yh, zero_division=0)),
            "specificity": spec,
        }

    groups = None
    if accessions is not None:
        groups = np.array([str(a).split("_")[0] + "_" + str(a).split("_")[1] if "_" in str(a) else str(a) for a in accessions])

    macro_auc_bs = []
    macro_f1_bs = []
    for _ in range(n_bootstrap):
        idx = _bootstrap_indices(len(true), groups=groups, rng=rng)
        macro_auc_bs.append(float(np.nanmean([_safe_auc(true[idx, j], pred[idx, j]) for j in range(pred.shape[1])])) )
        macro_f1_bs.append(float(np.nanmean([f1_score(true[idx, j].astype(int), y_hat[idx, j], zero_division=0) for j in range(pred.shape[1])])) )

    def _avg(metric):
        return float(np.nanmean([per_class[n][metric] for n in class_names]))

    macro_auc_bs = np.asarray(macro_auc_bs)
    macro_f1_bs = np.asarray(macro_f1_bs)
    return {
        "n_samples": int(len(true)),
        "n_classes": int(pred.shape[1]),
        "threshold": float(threshold),
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_unit": "patient_prefix" if groups is not None else "sample",
        "macro": {
            "auc": _avg("auc"),
            "auc_ci_lo": float(np.nanpercentile(macro_auc_bs, 2.5)),
            "auc_ci_hi": float(np.nanpercentile(macro_auc_bs, 97.5)),
            "acc": _avg("acc"),
            "f1": _avg("f1"),
            "f1_ci_lo": float(np.nanpercentile(macro_f1_bs, 2.5)),
            "f1_ci_hi": float(np.nanpercentile(macro_f1_bs, 97.5)),
            "precision": _avg("precision"),
            "sensitivity": _avg("sensitivity"),
            "specificity": _avg("specificity"),
        },
        "per_class": per_class,
    }


def _per_auc(pred, true):
    return {p: _safe_auc(true[:, j], pred[:, j]) for j, p in enumerate(PATHOLOGIES)}


def _routed_probs(global_lat, organ_lat, organ_valid, has_masks, pos_embs, neg_embs):
    B = global_lat.shape[0]
    routed_lat = global_lat.unsqueeze(1).expand(B, N_PATH, global_lat.shape[-1]).clone()
    n_routed = 0
    for pidx, pname in enumerate(PATHOLOGIES):
        organs = PATHOLOGY_FINE_ORGANS.get(pname, [])
        if not organs or pname in GLOBAL_EVAL_PATHOLOGIES:
            continue
        organ_idxs = [ORGAN_TO_INDEX[o] for o in organs if o in ORGAN_TO_INDEX]
        if not organ_idxs:
            continue
        valid = organ_valid[:, organ_idxs] & has_masks[:, None]
        any_valid = valid.any(dim=1)
        if not any_valid.any():
            continue
        weights = valid.float()
        pooled = (organ_lat[:, organ_idxs] * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        routed_lat[any_valid, pidx] = F.normalize(pooled[any_valid], dim=-1)
        n_routed += int(any_valid.sum().item())

    sims_pos = torch.einsum("bcl,cl->bc", F.normalize(routed_lat, dim=-1), pos_embs)
    sims_neg = torch.einsum("bcl,cl->bc", F.normalize(routed_lat, dim=-1), neg_embs)
    return F.softmax(torch.stack([sims_pos, sims_neg], dim=-1) / TEMPERATURE, dim=-1)[..., 0], n_routed


@torch.no_grad()
def evaluate(clip, tokenizer, device):
    clip.eval()
    qformer_module = getattr(clip, "qformer_module", None)
    use_anatomy_qformer = bool(getattr(clip, "use_anatomy_qformer", False))
    if qformer_module is not None:
        qformer_module.eval()
    pos_embs, neg_embs = encode_prompts(clip, tokenizer, device)
    val_ds = RACDatasetV4(
        DATA_VALID,
        REPORTS_VALID,
        LABELS_VALID,
        mask_root=MASK_VALID if os.path.isdir(MASK_VALID) else None,
        is_train=False,
        limit=VAL_LIMIT,
        fail_fast=FAIL_FAST,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=rac_collate, pin_memory=True)

    global_pred, routed_pred, all_true, all_accessions = [], [], [], []
    n_routed_total = 0
    for ct, _, _, labels, masks_fine, has_masks, accessions in tqdm.tqdm(val_loader, desc="Eval"):
        ct = ct.to(device, non_blocking=True)
        masks_fine = masks_fine.to(device, non_blocking=True)
        has_masks = has_masks.to(device, non_blocking=True)
        feat_map = clip.visual_transformer.forward_spatial(ct)
        if qformer_module is not None:
            # Q-Former replaces the global to_visual_latent path. The 32
            # learnable queries cross-attend to the layer4 feature map;
            # ``pool="mean"`` averages the 32 tokens into a single 768-d
            # vector that we L2-normalize as the visual latent.
            if use_anatomy_qformer:
                pooled = qformer_module(feat_map, masks_fine, has_masks)
            else:
                pooled = qformer_module(feat_map)
            global_lat = F.normalize(pooled, dim=-1)
        else:
            raw = clip.visual_transformer.global_pool(feat_map)
            global_lat = F.normalize(clip.to_visual_latent(raw), dim=-1)
        gp = prompt_probs(global_lat, pos_embs, neg_embs)

        if ORGAN_LABELS and masks_fine.numel() > 0:
            organ_raw, organ_valid = clip.visual_transformer.pool_organ_masks(feat_map, masks_fine, ORGAN_LABELS)
            B, O, C = organ_raw.shape
            organ_lat = F.normalize(clip.to_visual_latent(organ_raw.reshape(B * O, C)).reshape(B, O, -1), dim=-1)
            rp, n_routed = _routed_probs(global_lat, organ_lat, organ_valid, has_masks, pos_embs, neg_embs)
            n_routed_total += n_routed
        else:
            rp = gp

        global_pred.append(gp.float().cpu().numpy())
        routed_pred.append(rp.float().cpu().numpy())
        all_true.append(labels.numpy())
        all_accessions.extend(accessions)

    pred_global = np.concatenate(global_pred)
    pred_routed = np.concatenate(routed_pred)
    true = np.concatenate(all_true).astype(np.int32)
    accessions = np.asarray(all_accessions)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    bundle_global = compute_metric_bundle(pred_global, true, PATHOLOGIES, accessions=accessions, threshold=THRESHOLD, n_bootstrap=N_BOOTSTRAP)
    bundle_routed = compute_metric_bundle(pred_routed, true, PATHOLOGIES, accessions=accessions, threshold=THRESHOLD, n_bootstrap=N_BOOTSTRAP)
    with open(f"{RESULTS_DIR}/metric_bundle_global.json", "w") as f:
        json.dump(bundle_global, f, indent=2)
    with open(f"{RESULTS_DIR}/metric_bundle_routed.json", "w") as f:
        json.dump(bundle_routed, f, indent=2)

    pd.DataFrame(pred_global, columns=PATHOLOGIES).to_csv(f"{RESULTS_DIR}/predicted_labels_global.csv", index=False)
    pd.DataFrame(pred_routed, columns=PATHOLOGIES).to_csv(f"{RESULTS_DIR}/predicted_labels_routed.csv", index=False)

    np.savez_compressed(
        f"{RESULTS_DIR}/predictions.npz",
        pred=pred_global.astype(np.float32),
        true=true,
        pathologies=np.array(PATHOLOGIES),
        accessions=accessions,
        mode=np.array("global_primary"),
    )
    np.savez_compressed(
        f"{RESULTS_DIR}/predictions_routed.npz",
        pred=pred_routed.astype(np.float32),
        true=true,
        pathologies=np.array(PATHOLOGIES),
        accessions=accessions,
        mode=np.array("ts_mask_routed_secondary"),
    )

    auc_global = _per_auc(pred_global, true)
    auc_routed = _per_auc(pred_routed, true)
    with open(f"{RESULTS_DIR}/auc_scores.txt", "w") as f:
        f.write("Primary metric: mask-free global prompt AUC\n")
        f.write("Secondary metric: TS-mask-routed AUC when masks are available\n\n")
        f.write(f"{'pathology':45s}: {'g_auc':>8s} {'r_auc':>8s} {'g_acc':>8s} {'r_acc':>8s} {'g_prec':>8s} {'r_prec':>8s} {'g_f1':>8s} {'r_f1':>8s}\n")
        for p in PATHOLOGIES:
            bg = bundle_global['per_class'].get(p, {})
            br = bundle_routed['per_class'].get(p, {})
            f.write(f"{p:45s}: {auc_global[p]:8.4f} {auc_routed[p]:8.4f}"
                    f" {bg.get('acc', float('nan')):8.4f} {br.get('acc', float('nan')):8.4f}"
                    f" {bg.get('precision', float('nan')):8.4f} {br.get('precision', float('nan')):8.4f}"
                    f" {bg.get('f1', float('nan')):8.4f} {br.get('f1', float('nan')):8.4f}\n")
        f.write(f"\n{'Mean AUC':45s}: {bundle_global['macro']['auc']:8.4f} {bundle_routed['macro']['auc']:8.4f}\n")
        f.write(f"{'Mean Acc':45s}: {bundle_global['macro']['acc']:8.4f} {bundle_routed['macro']['acc']:8.4f}\n")
        f.write(f"{'Mean Precision':45s}: {bundle_global['macro']['precision']:8.4f} {bundle_routed['macro']['precision']:8.4f}\n")
        f.write(f"{'Mean F1':45s}: {bundle_global['macro']['f1']:8.4f} {bundle_routed['macro']['f1']:8.4f}\n")
        f.write(f"{'Global AUC 95% CI':45s}: [{bundle_global['macro']['auc_ci_lo']:.4f}, {bundle_global['macro']['auc_ci_hi']:.4f}]\n")
        f.write(f"{'Routed AUC 95% CI':45s}: [{bundle_routed['macro']['auc_ci_lo']:.4f}, {bundle_routed['macro']['auc_ci_hi']:.4f}]\n")
        f.write(f"{'Global F1 95% CI':45s}: [{bundle_global['macro']['f1_ci_lo']:.4f}, {bundle_global['macro']['f1_ci_hi']:.4f}]\n")
        f.write(f"{'Routed F1 95% CI':45s}: [{bundle_routed['macro']['f1_ci_lo']:.4f}, {bundle_routed['macro']['f1_ci_hi']:.4f}]\n")
        f.write(f"{'Routed sample-class decisions':45s}: {n_routed_total}\n")

    print(f"[eval] Global  AUC={bundle_global['macro']['auc']:.4f} CI=[{bundle_global['macro']['auc_ci_lo']:.4f}, {bundle_global['macro']['auc_ci_hi']:.4f}]"
          f"  Acc={bundle_global['macro']['acc']:.4f}  Prec={bundle_global['macro']['precision']:.4f}  F1={bundle_global['macro']['f1']:.4f}")
    print(f"[eval] Routed  AUC={bundle_routed['macro']['auc']:.4f} CI=[{bundle_routed['macro']['auc_ci_lo']:.4f}, {bundle_routed['macro']['auc_ci_hi']:.4f}]"
          f"  Acc={bundle_routed['macro']['acc']:.4f}  Prec={bundle_routed['macro']['precision']:.4f}  F1={bundle_routed['macro']['f1']:.4f}")
    print(f"[eval] Saved to {RESULTS_DIR}")


def main():
    assert CKPT, "EVAL_CKPT env var required"
    assert RESULTS_DIR, "EVAL_RESULTS_DIR env var required"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip, tokenizer = build_model_for_eval(CKPT)
    clip = clip.to(device)
    # Side-attached Q-Former family modules need their own .to(device) since
    # they are not registered submodules of the CTCLIP container.
    for attr in ("qformer_module", "itm_module", "lm_bridge"):
        mod = getattr(clip, attr, None)
        if mod is not None:
            setattr(clip, attr, mod.to(device))
    evaluate(clip, tokenizer, device)


if __name__ == "__main__":
    main()
