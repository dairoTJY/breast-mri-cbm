#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import json
import csv
import math
import random
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score, roc_curve

# Reuse the unified dataset
from ispy2_dataset_finetune import ISPY2Dataset

# ---- safe import for train_resnet.py (avoid sys.argv side effects) ----
import sys
_argv_bak = sys.argv
sys.argv = [sys.argv[0]]
try:
    from train_resnet import ResNet3DClassifier
finally:
    sys.argv = _argv_bak
# ----------------------------------------------------------------------


# ----------------------------
# Utils (seed, split, logging)
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# def worker_init_fn(worker_id: int, base_seed: int):
#     np.random.seed(base_seed + worker_id)
#     random.seed(base_seed + worker_id)
#     torch.manual_seed(base_seed + worker_id)

def worker_init_fn(worker_id: int, seed: int):
    import random
    import numpy as np
    import torch
    from torch.utils.data import get_worker_info
    s = int(seed) + int(worker_id) * 10000

    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

    # ---- CRITICAL: seed MONAI random transforms' internal RNG ----
    wi = get_worker_info()
    if wi is not None:
        ds = wi.dataset
        # Subset -> unwrap
        try:
            from torch.utils.data import Subset
            if isinstance(ds, Subset):
                ds = ds.dataset
        except Exception:
            pass

        # Some pipelines may wrap again (Subset of Subset)
        while hasattr(ds, "dataset"):
            # avoid infinite loop on dataset objects that also have dataset attr for other reasons
            if ds.__class__.__name__ == "Subset":
                ds = ds.dataset
            else:
                break

        if hasattr(ds, "geom_aug") and ds.geom_aug is not None:
            ds.geom_aug.set_random_state(seed=s)
        if hasattr(ds, "int_aug") and ds.int_aug is not None:
            ds.int_aug.set_random_state(seed=s)

def custom_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_split_pids(split_json: str) -> Tuple[List[str], List[str], List[str], Dict[str, Any]]:
    with open(split_json, "r") as f:
        s = json.load(f)
    train_pids = s.get("train_pids", [])
    val_pids = s.get("val_pids", [])
    test_pids = s.get("test_pids", [])
    return train_pids, val_pids, test_pids, s


def pids_to_indices(pids: List[str], all_pids: List[str], strict: bool = True) -> List[int]:
    pid2idx = {p: i for i, p in enumerate(all_pids)}
    missing = [p for p in pids if p not in pid2idx]
    if missing and strict:
        raise RuntimeError(f"[split] Missing {len(missing)} pids not found in dataset json. Example: {missing[:5]}")
    return [pid2idx[p] for p in pids if p in pid2idx]


def init_csv_logger(path_csv: str, header: List[str]):
    ensure_dir(os.path.dirname(path_csv))
    exists = os.path.exists(path_csv)
    f = open(path_csv, "a", newline="")
    w = csv.writer(f)
    if not exists:
        w.writerow(header)
        f.flush()
    return f, w


# ----------------------------
# Model: Route A projection CBM
# ----------------------------
class PCBMRouteA(nn.Module):
    """
    Frozen 3D ResNet backbone -> pooled feature h -> linear projection Wf -> z (Dt)
    -> cosine with fixed concept embedding E (K x Dt) -> concept scores c (K)
    -> optional alpha gate -> c_eff -> linear head -> logit
    """
    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        concept_emb: torch.Tensor,  # [K, Dt]
        use_alpha: bool = False,
        alpha_hidden: int = 256,
        clinical_dim: int = 0,
        clinical_hidden: int = 128,
        use_linear_clinical_head: bool = False,
        modulation_input_dim: int = 0,
        modulation_hidden: int = 64,
        modulation_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim

        # Concept embedding buffer (fixed)
        assert concept_emb.ndim == 2, f"concept_emb must be [K,Dt], got {tuple(concept_emb.shape)}"
        self.K, self.Dt = concept_emb.shape
        self.register_buffer("E", F.normalize(concept_emb.float(), dim=1), persistent=False)

        # Projection Wf
        self.Wf = nn.Linear(feat_dim, self.Dt, bias=False)

        self.use_clinical = clinical_dim > 0
        self.clinical_dim = int(clinical_dim)
        self.use_linear_clinical_head = bool(use_linear_clinical_head)
        self.use_clinical_modulation = int(modulation_input_dim) > 0
        self.modulation_input_dim = int(modulation_input_dim)
        self.modulation_indices = list(modulation_indices) if modulation_indices is not None else []
        if self.use_clinical:
            if self.use_linear_clinical_head:
                self.clinical_net = None
                head_in_dim = self.K + self.clinical_dim
            else:
                self.clinical_net = nn.Sequential(
                    nn.LayerNorm(clinical_dim),
                    nn.Linear(clinical_dim, clinical_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(clinical_hidden, clinical_hidden),
                    nn.ReLU(inplace=True),
                )
                head_in_dim = self.K + clinical_hidden
        else:
            self.clinical_net = None
            head_in_dim = self.K

        if self.use_clinical_modulation:
            self.modulation_net = nn.Sequential(
                nn.LayerNorm(self.modulation_input_dim),
                nn.Linear(self.modulation_input_dim, modulation_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(modulation_hidden, self.K),
                nn.Sigmoid(),
            )
        else:
            self.modulation_net = None

        # Head g
        self.head = nn.Linear(head_in_dim, 1, bias=True)

        # Optional alpha gate (AdaCBM-style)
        self.use_alpha = bool(use_alpha)
        if self.use_alpha:
            self.alpha_net = nn.Sequential(
                nn.Linear(feat_dim, alpha_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(alpha_hidden, self.K),
                nn.Sigmoid()
            )

    def build_head_input(self, c_eff: torch.Tensor, clinical_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_clinical:
            if clinical_feat is None:
                raise ValueError("clinical_feat is required when use_clinical=True")
            clin = clinical_feat.to(c_eff.device, c_eff.dtype)
            if self.use_linear_clinical_head:
                clin_emb = clin
            else:
                clin_emb = self.clinical_net(clin)
            return torch.cat([c_eff, clin_emb], dim=1)
        return c_eff

    def classify_from_concepts(self, c_eff: torch.Tensor, clinical_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        head_in = self.build_head_input(c_eff, clinical_feat)
        return self.head(head_in).squeeze(-1)

    def apply_clinical_modulation(self, c_eff: torch.Tensor, clinical_feat: Optional[torch.Tensor] = None):
        if not self.use_clinical_modulation:
            return c_eff, None
        if clinical_feat is None:
            raise ValueError("clinical_feat is required when use_clinical_modulation=True")
        mod_in = clinical_feat.to(c_eff.device, c_eff.dtype)
        if self.modulation_indices:
            mod_in = mod_in[:, self.modulation_indices]
        gate = self.modulation_net(mod_in)
        c_mod = c_eff * (1.0 + gate)
        return c_mod, gate

    def forward(
        self,
        img: torch.Tensor,
        msk: torch.Tensor,
        clinical_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # backbone forward: your ResNet3DClassifier expects 3ch input, you already provide (dce1,dce2,mask)
        # Here, img is already stacked by dataset as expected; msk used in your baseline stacking pipeline
        # In your current training loop, you pass (img, msk) to model; we follow that signature.

        # backbone returns logits in its own classifier; we need features instead.
        # In your train_resnet.py, ResNet3DClassifier exposes backbone or features via forward_features.
        # This script keeps your original "dummy forward to infer feat_dim" logic and uses
        # a safe feature extraction wrapper below.
        h = self._extract_feat(img, msk)  # [B, feat_dim]

        z = self.Wf(h)  # [B, Dt]
        z = F.normalize(z, dim=1)

        # cosine similarity to E (K x Dt): c = z @ E^T
        c = torch.matmul(z, self.E.t())  # [B, K]

        if self.use_alpha:
            alpha = self.alpha_net(h)  # [B,K] in (0,1)
            c_eff = alpha * c
        else:
            alpha = None
            c_eff = c

        c_head, gate = self.apply_clinical_modulation(c_eff, clinical_feat)
        logit = self.classify_from_concepts(c_head, clinical_feat)  # [B]
        return {"h": h, "z": z, "c": c, "c_eff": c_eff, "c_head": c_head, "alpha": alpha, "gate": gate, "logit": logit}

    def _extract_feat(self, img: torch.Tensor, msk: torch.Tensor) -> torch.Tensor:
        """
        Keep consistent with your original script:
        - build 3-channel input by concatenating img and mask appropriately if needed
        - call the ResNet3DClassifier's feature extractor path
        """
        # In your dataset, imgs already include 3 channels (dce1,dce2,mask) typically.
        # Your training code passes imgs=batch[6], msks=batch[7]; in baseline you may concatenate.
        # Here we follow your existing behavior: if img has C==2 and msk exists, concat.
        if img.ndim == 5 and img.shape[1] == 2:
            x = torch.cat([img, msk], dim=1)
        else:
            x = img

        # Try common feature extraction APIs
        if hasattr(self.backbone, "feature_extractor"):
            feat = self.backbone.feature_extractor(x, msk)
        elif hasattr(self.backbone, "forward_features"):
            feat = self.backbone.forward_features(x)
        elif hasattr(self.backbone, "backbone"):
            # Some wrappers store the actual backbone
            bb = getattr(self.backbone, "backbone")
            feat = bb(x)
        else:
            # Fallback: try to remove classifier by checking for 'avgpool' / 'fc'
            feat = self.backbone(x)

        # feat could be [B, C] or [B, C, ...]; pool if needed
        if feat.ndim > 2:
            feat = torch.flatten(F.adaptive_avg_pool3d(feat, 1), 1)
        return feat


# ----------------------------
# Helpers: batch unpack & metrics
# ----------------------------
def unpack_batch(batch, device, need_pid: bool = False):
    """
    dataset tuple layout:
      0 index_tensor
      1 t2_tensor
      2 dwi_tensor
      3 dce_tensor
      4 label_tensor
      5 mask_tensor
      6 img_aug (3ch)   <-- used
      7 mask_aug (1ch)  <-- used (may be None in some edge cases)
      8 clinical_tensor
      9 pid (optional, if return_pid=True)
    """
    imgs = batch[6].to(device, non_blocking=True)

    msks_raw = batch[7]
    if msks_raw is None:
        # Fallback to non-aug mask if augmentation mask missing
        msks_raw = batch[5]
    if msks_raw is None:
        # Last resort: zeros mask
        msks = torch.zeros((imgs.shape[0], 1, imgs.shape[2], imgs.shape[3], imgs.shape[4]),
                           device=device, dtype=imgs.dtype)
    else:
        msks = msks_raw.to(device, non_blocking=True).to(imgs.dtype)

    labs = torch.as_tensor(batch[4], device=device).float().view(-1)
    clinical = batch[8].to(device, non_blocking=True) if len(batch) > 8 and torch.is_tensor(batch[8]) else None
    if clinical is not None and clinical.ndim >= 2 and clinical.shape[1] == 0:
        clinical = None
    pid = batch[-1] if need_pid else None
    return imgs, msks, labs, clinical, pid


def resolve_modulation_indices(feature_names: Optional[List[str]], requested: Optional[List[str]]) -> List[int]:
    if not feature_names or not requested:
        return []
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    resolved: List[int] = []

    def add_name(name: str):
        if name not in name_to_idx:
            raise KeyError(f"Unknown clinical feature for modulation: {name}")
        idx = name_to_idx[name]
        if idx not in resolved:
            resolved.append(idx)

    for item in requested:
        key = str(item).strip()
        if not key:
            continue
        if key == "menopause":
            for n in ["meno_pre", "meno_peri", "meno_post", "meno_na_gt50", "meno_na_lt50", "meno_unknown"]:
                add_name(n)
        elif key == "race":
            for n in ["race_white", "race_black", "race_asian", "race_pacific_islander", "race_native_american", "race_unknown"]:
                add_name(n)
        else:
            add_name(key)
    return resolved


def compute_basic_metrics(labels: np.ndarray, probs: np.ndarray, thr: float = 0.5) -> Dict[str, Any]:
    labels = labels.astype(int).ravel()
    probs = probs.astype(float).ravel()
    preds = (probs >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    acc = accuracy_score(labels, preds)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5
    return {
        "auc": float(auc), "f1": float(f1), "acc": float(acc), "sens": float(sens), "spec": float(spec),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }


def compute_youden_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = labels.astype(int).ravel()
    probs = probs.astype(float).ravel()
    if len(np.unique(labels)) <= 1:
        return 0.5
    fpr, tpr, thr = roc_curve(labels, probs)
    youden = tpr - fpr
    yi = int(np.argmax(youden))
    return float(thr[yi])


@torch.no_grad()
def validate(model: PCBMRouteA, loader, criterion, device, split_name: str = "Val", fixed_thr: float = None, compute_youden: bool = True, return_metrics: bool = False):
    model.eval()
    total_loss, valid_steps = 0.0, 0
    all_probs, all_labels = [], []

    pbar = tqdm(loader, desc=f"{split_name}", leave=False)
    for batch in pbar:
        if batch is None:
            continue
        imgs, msks, labs, clinical, pid = unpack_batch(batch, device, need_pid=False)

        out = model(imgs, msks, clinical)

        # 统一成 1D
        logit = out["logit"].view(-1)
        labs  = labs.view(-1)  # 如果 unpack_batch 已经 float().view(-1)，这行可留可不留

        loss = criterion(logit, labs)

        total_loss += float(loss.item())
        valid_steps += 1

        probs = torch.sigmoid(logit).detach().cpu().numpy()   # shape (B,)
        all_probs.extend(probs.tolist())                      # 变成 python float list
        all_labels.extend(labs.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, valid_steps)
    probs = np.asarray(all_probs).ravel()
    labels = np.asarray(all_labels).astype(int).ravel()

    m05 = compute_basic_metrics(labels, probs, thr=0.5)
    print(f" {split_name} @0.5  AUC {m05['auc']:.4f} | F1 {m05['f1']:.4f} | Acc {m05['acc']:.4f} | "
          f"Sens {m05['sens']:.4f} | Spec {m05['spec']:.4f} | ConfMat [[TN {m05['tn']} FP {m05['fp']}] [FN {m05['fn']} TP {m05['tp']}]]")

    # Threshold-based report:
    # - If fixed_thr is provided, report metrics at that fixed threshold (recommended for Test: use Val-derived thr).
    # - Optionally also compute Youden on this split (useful for Val).
    youden_thr = None

    if fixed_thr is not None:
        mf = compute_basic_metrics(labels, probs, thr=float(fixed_thr))
        print(f" {split_name} (FixedThr) Thr {float(fixed_thr):.3f} | Sens {mf['sens']:.4f} | Spec {mf['spec']:.4f} "
              f"| ConfMat [[TN {mf['tn']} FP {mf['fp']}] [FN {mf['fn']} TP {mf['tp']}]]")
        youden_thr = float(fixed_thr)

    if compute_youden:
        yt = compute_youden_threshold(labels, probs)
        my = compute_basic_metrics(labels, probs, thr=yt)
        if len(np.unique(labels)) > 1:
            print(f" {split_name} (Youden) Thr {yt:.3f} | Sens {my['sens']:.4f} | Spec {my['spec']:.4f} "
                  f"| ConfMat [[TN {my['tn']} FP {my['fp']}] [FN {my['fn']} TP {my['tp']}]]")
        else:
            print(f" {split_name} (Youden) skipped: only one class.")
        if fixed_thr is None:
            youden_thr = float(yt)

    if youden_thr is None:
        youden_thr = 0.5

    metrics = {
        "thr_used": float(youden_thr),
        "m@0.5": m05,
    }
    if fixed_thr is not None:
        metrics["m@fixed_thr"] = compute_basic_metrics(labels, probs, thr=float(fixed_thr))
        metrics["fixed_thr"] = float(fixed_thr)
    if return_metrics:
        return avg_loss, float(m05["auc"]), float(youden_thr), metrics
    return avg_loss, float(m05["auc"]), float(youden_thr)


# ----------------------------
# Offline evaluation: collect concepts / interventions / TTA reliability
# ----------------------------
@torch.no_grad()
def collect_concepts_and_probs(model: PCBMRouteA, loader, device, use_c_eff: bool = True, need_pid: bool = True):
    model.eval()
    Cs, logits_list, probs_list, Ys, PIDs, Clinicals = [], [], [], [], [], []
    for batch in tqdm(loader, desc="Collect", leave=False):
        if batch is None:
            continue
        imgs, msks, labs, clinical, pid = unpack_batch(batch, device, need_pid=need_pid)
        out = model(imgs, msks, clinical)
        if use_c_eff:
            c = out["c_head"] if model.use_clinical_modulation else out["c_eff"]
        else:
            c = out["c"]
        logit = out["logit"]
        prob = torch.sigmoid(logit)

        Cs.append(c.detach().cpu().numpy())
        logits_list.append(logit.detach().cpu().numpy())
        probs_list.append(prob.detach().cpu().numpy())
        Ys.append(labs.detach().cpu().numpy())
        if model.use_clinical:
            if clinical is None:
                raise RuntimeError("Model expects clinical features but batch clinical is None.")
            Clinicals.append(clinical.detach().cpu().numpy())
        if need_pid:
            if isinstance(pid, (list, tuple)):
                PIDs.extend(list(pid))
            else:
                # pid could be tensor of strings? usually list[str]
                try:
                    PIDs.extend(list(pid))
                except Exception:
                    PIDs.append(pid)

    C = np.concatenate(Cs, axis=0) if Cs else np.zeros((0, model.K), dtype=np.float32)
    logits = np.concatenate(logits_list, axis=0).ravel() if logits_list else np.zeros((0,), dtype=np.float32)
    probs = np.concatenate(probs_list, axis=0).ravel() if probs_list else np.zeros((0,), dtype=np.float32)
    Y = np.concatenate(Ys, axis=0).astype(int).ravel() if Ys else np.zeros((0,), dtype=np.int64)
    clinical_np = np.concatenate(Clinicals, axis=0) if Clinicals else None
    return C, logits, probs, Y, PIDs, clinical_np


@torch.no_grad()
def head_forward(model: PCBMRouteA, c_np: np.ndarray, device, clinical_np: Optional[np.ndarray] = None):
    c = torch.from_numpy(c_np).to(device=device, dtype=torch.float32)
    clinical = None
    if clinical_np is not None:
        clinical = torch.from_numpy(clinical_np).to(device=device, dtype=torch.float32)
    c_for_head = c
    if model.use_clinical_modulation:
        c_for_head, _ = model.apply_clinical_modulation(c, clinical)
    logit = model.classify_from_concepts(c_for_head, clinical).squeeze(-1)
    prob = torch.sigmoid(logit)
    return logit.detach().cpu().numpy().ravel(), prob.detach().cpu().numpy().ravel()


def run_faithfulness(
    model: PCBMRouteA,
    C0: np.ndarray,
    clinical0: Optional[np.ndarray],
    probs0: np.ndarray,
    labels: np.ndarray,
    pids: List[str],
    device,
    out_dir: str,
    topm_list: List[int] = (1, 3, 5),
    do_random_topm: bool = True,
    random_repeats: int = 5,
    seed: int = 20,
    p_change_eps: float = 0.05,
):
    """Faithfulness evaluation used in the paper (minimal, table-only).

    We **only** report top-m concept ablation (zeroing) with an optional random-m sanity check.
    Metrics are threshold-free:
      - mean_abs_delta_prob = mean_i |p1_i - p0_i|
      - auc0 / auc1 / delta_auc
      - p_change_rate_eps = fraction_i (|p1_i - p0_i| > eps)

    Notes
    -----
    - top-m is computed per instance by sorting |a_{i,k}| = |w_k * c_{i,k}|.
    - interventions are applied on the head input (c_eff when alpha is on, else c).
    """
    ensure_dir(out_dir)

    C0 = np.asarray(C0, dtype=np.float32)
    probs0 = np.asarray(probs0, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    N, K = C0.shape

    # Baseline AUC
    try:
        auc0 = float(roc_auc_score(labels, probs0)) if len(np.unique(labels)) > 1 else float("nan")
    except Exception:
        auc0 = float("nan")

    # Contribution magnitude per instance
    w_head = model.head.weight.detach().cpu().numpy().reshape(-1)[:model.K]  # [K]
    contrib = C0 * w_head[None, :]  # [N,K]

    rng = np.random.default_rng(int(seed))
    rows = []

    for m in list(topm_list):
        m = int(m)
        # Per-sample top-m indices
        idx = np.argsort(np.abs(contrib), axis=1)[:, ::-1][:, :m]
        C1 = C0.copy()
        for i in range(N):
            C1[i, idx[i]] = 0.0

        _, prob1 = head_forward(model, C1, device, clinical0)
        prob1 = np.asarray(prob1, dtype=np.float32).reshape(-1)
        dprob = prob1 - probs0

        try:
            auc1 = float(roc_auc_score(labels, prob1)) if len(np.unique(labels)) > 1 else float("nan")
        except Exception:
            auc1 = float("nan")
        delta_auc = float(auc1 - auc0) if (not math.isnan(auc0) and not math.isnan(auc1)) else float("nan")

        row = {
            "m": m,
            "mean_abs_delta_prob": float(np.mean(np.abs(dprob))),
            "auc0": auc0,
            "auc1": auc1,
            "delta_auc": delta_auc,
            f"p_change_rate_{p_change_eps:g}": float(np.mean(np.abs(dprob) > float(p_change_eps))),
        }

        # Random-m sanity check (optional)
        if do_random_topm and int(random_repeats) > 0:
            stats = []
            for _ in range(int(random_repeats)):
                C1r = C0.copy()
                for i in range(N):
                    ridx = rng.choice(K, size=m, replace=False)
                    C1r[i, ridx] = 0.0
                _, prob1r = head_forward(model, C1r, device, clinical0)
                prob1r = np.asarray(prob1r, dtype=np.float32).reshape(-1)
                dprobr = prob1r - probs0

                try:
                    auc1r = float(roc_auc_score(labels, prob1r)) if len(np.unique(labels)) > 1 else float("nan")
                except Exception:
                    auc1r = float("nan")
                delta_aucr = float(auc1r - auc0) if (not math.isnan(auc0) and not math.isnan(auc1r)) else float("nan")

                stats.append([
                    float(np.mean(np.abs(dprobr))),
                    float(np.mean(np.abs(dprobr) > float(p_change_eps))),
                    auc1r,
                    delta_aucr,
                ])
            stats = np.asarray(stats, dtype=float)

            row.update({
                "rand_repeats": int(random_repeats),
                "rand_mean_abs_delta_prob": float(np.nanmean(stats[:, 0])),
                f"rand_p_change_rate_{p_change_eps:g}": float(np.nanmean(stats[:, 1])),
                "rand_auc1": float(np.nanmean(stats[:, 2])),
                "rand_delta_auc": float(np.nanmean(stats[:, 3])),
            })

        rows.append(row)

    # Write CSV
    out_csv = os.path.join(out_dir, "faithfulness_topm_summary.csv")
    pcol = f"p_change_rate_{p_change_eps:g}"
    rpcol = f"rand_p_change_rate_{p_change_eps:g}"
    base_cols = ["m", "mean_abs_delta_prob", "auc0", "auc1", "delta_auc", pcol]
    rand_cols = ["rand_repeats", "rand_mean_abs_delta_prob", rpcol, "rand_auc1", "rand_delta_auc"]
    cols = base_cols + (rand_cols if do_random_topm else [])
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])

    print(f"[FAITH] saved: {out_csv}")


def run_reliability_tta(
    model: PCBMRouteA,
    tta_dataset,
    tta_indices: List[int],
    batch_size: int,
    num_workers: int,
    pin_mem: bool,
    base_labels: np.ndarray,
    base_pids: List[str],
    device,
    out_dir: str,
    tta_T: int = 10,
    seed: int = 20,
    use_c_eff: bool = True,
    coverages: List[int] = (50, 70, 90, 100),
    val_thr: float = 0.5,
    reliability_rank: str = "concept_R",  # concept_R | confidence | prob_std
    report_valthr: bool = False,
):
    """
    TTA reliability (paper-ready):
    - IMPORTANT: For MONAI random transforms, each DataLoader worker has its own RNG state.
      Simply calling set_seed() in the main process is NOT sufficient when num_workers>0.
      Therefore, we rebuild the DataLoader for every TTA pass with a different worker_init seed.
    - Each pass t uses a different base_seed = seed + 1000 + t.
    - shuffle=False so that pid order is consistent across passes.

    Outputs:
      - reliability_percase.csv: pid, label, R, U, prob_mean, prob_std
      - coverage_performance.csv: coverage vs metrics at thr=0.5 and thr=val_thr (val-derived)
    """
    ensure_dir(out_dir)
    N = len(base_pids)

    C_all: List[np.ndarray] = []
    P_all: List[np.ndarray] = []

    # Build subset once
    tta_ds = Subset(tta_dataset, tta_indices)

    # imgs0 = None

    for t in range(tta_T):
        base_seed = int(seed + 1000 + t)

        # Rebuild loader so worker RNG seeds change each pass (critical!)
        tta_loader = DataLoader(
            tta_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem,
            collate_fn=custom_collate_fn,
            worker_init_fn=lambda wid, bs=base_seed: worker_init_fn(wid, bs),
            persistent_workers=False,
        )

        # Also seed main process for completeness (affects single-worker / any main-thread randomness)
        set_seed(base_seed)


        C_t, _, p_t, _, pids_t, _ = collect_concepts_and_probs(
            model, tta_loader, device, use_c_eff=use_c_eff, need_pid=True
        )

        if len(pids_t) != N:
            raise RuntimeError(
                f"[TTA] pid count mismatch: pass {t} got {len(pids_t)} vs base {N}. "
                f"Ensure shuffle=False and the same Subset indices."
            )
        # Ensure consistent order; if ever mismatched, you should sort by pid, but we enforce order equality here
        if list(map(str, pids_t)) != list(map(str, base_pids)):
            raise RuntimeError(
                f"[TTA] pid order mismatch at pass {t}. "
                f"This indicates nondeterministic ordering; set shuffle=False and avoid dataset-side shuffling."
            )

        C_all.append(C_t.astype(np.float32))
        P_all.append(p_t.astype(np.float32))

    C_stack = np.stack(C_all, axis=0)  # [T,N,K]
    P_stack = np.stack(P_all, axis=0)  # [T,N]

    std_k = C_stack.std(axis=0)        # [N,K]
    np.save(os.path.join(out_dir, "concept_std_NxK.npy"), std_k.astype(np.float32))

    U = std_k.mean(axis=1)             # [N]
    R = np.exp(-U)                     # [N]

    prob_mean = P_stack.mean(axis=0)
    prob_std = P_stack.std(axis=0)

    # Save per-case reliability
    percase_csv = os.path.join(out_dir, "reliability_percase.csv")
    with open(percase_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pid", "label", "R", "U", "prob_mean", "prob_std"])
        for i in range(N):
            w.writerow([base_pids[i], int(base_labels[i]), float(R[i]), float(U[i]), float(prob_mean[i]), float(prob_std[i])])


    # Coverage-performance (selective prediction)
    reliability_rank = str(reliability_rank)
    if reliability_rank not in ("concept_R", "confidence", "prob_std"):
        raise ValueError(f"Unknown reliability_rank: {reliability_rank}. Use concept_R|confidence|prob_std")

    # Higher score = keep first
    if reliability_rank == "concept_R":
        score = R  # high R -> reliable
    elif reliability_rank == "confidence":
        score = np.abs(prob_mean - 0.5)  # farther from boundary -> reliable
    else:  # prob_std
        score = -prob_std  # lower std -> higher score

    order = np.argsort(-score)  # high to low
    cov_rows = []
    for cov in coverages:
        keep_n = int(math.ceil(N * cov / 100.0))
        idx = order[:keep_n]
        y_sub = base_labels[idx]
        p_sub = prob_mean[idx]

        m05 = compute_basic_metrics(y_sub, p_sub, thr=0.5)
        if report_valthr:
            my = compute_basic_metrics(y_sub, p_sub, thr=val_thr)
            cov_rows.append([cov, keep_n,
                             m05["auc"], m05["f1"], m05["sens"], m05["spec"],
                             my["f1"], my["sens"], my["spec"], my["acc"]])
        else:
            cov_rows.append([cov, keep_n,
                             m05["auc"], m05["f1"], m05["sens"], m05["spec"]])

    cov_csv = os.path.join(out_dir, "coverage_performance.csv")
    with open(cov_csv, "w", newline="") as f:
        w = csv.writer(f)
        if report_valthr:
            w.writerow(["coverage_pct", "n_keep",
                        "auc@0.5", "f1@0.5", "sens@0.5", "spec@0.5",
                        "f1@valThr", "sens@valThr", "spec@valThr", "acc@valThr"])
        else:
            w.writerow(["coverage_pct", "n_keep",
                        "auc@0.5", "f1@0.5", "sens@0.5", "spec@0.5"])
        w.writerows(cov_rows)

    print(f"[RELI] saved: {percase_csv} and {cov_csv}")



    # ----------------------------
# Build datasets / loaders
# ----------------------------
def build_datasets_and_loaders(args):
    """
    Builds:
    - full_dataset_train (train mode, possibly aug)
    - full_dataset_eval  (eval mode, no aug, return_pid=True if args.need_pid)
    - train/val/test subsets + loaders
    """
    pin_mem = (args.device.startswith("cuda") and torch.cuda.is_available())

    # Sizes (use CLI)
    target_img_size_dce = tuple(args.img_size_dce)
    target_img_size_t2  = tuple(args.img_size_t2) if args.img_size_t2 is not None else target_img_size_dce
    target_img_size_dwi = tuple(args.img_size_dwi) if args.img_size_dwi is not None else target_img_size_dce

    # Train dataset (keep original behavior)
    full_dataset_train = ISPY2Dataset(
        json_path=args.json_path,
        img_size_dce=target_img_size_dce,
        img_size_t2=target_img_size_t2,
        img_size_dwi=target_img_size_dwi,
        repeat_channels=1,
        return_pid=False,
        is_train=True,
        use_aug=getattr(args, 'use_aug', True),
    )

    # Eval dataset (no aug). In eval mode we need pid; in train mode not required,
    # but we can still build return_pid=True without hurting.
    full_dataset_eval = ISPY2Dataset(
        json_path=args.json_path,
        img_size_dce=target_img_size_dce,
        img_size_t2=target_img_size_t2,
        img_size_dwi=target_img_size_dwi,
        repeat_channels=1,
        return_pid=True,
        is_train=False,
        use_aug=False,
    )

    # Split
    train_pids, val_pids, test_pids, _ = load_split_pids(args.split_json)
    print(f"[INFO] Split: train/val/test = {len(train_pids)}/{len(val_pids)}/{len(test_pids)}")

    train_indices = pids_to_indices(train_pids, full_dataset_train.pids, strict=args.strict_split_check)
    val_indices   = pids_to_indices(val_pids,   full_dataset_train.pids, strict=args.strict_split_check)
    test_indices  = pids_to_indices(test_pids,  full_dataset_train.pids, strict=args.strict_split_check) if len(test_pids) else []

    train_ds = Subset(full_dataset_train, train_indices)
    val_ds   = Subset(full_dataset_eval,  val_indices)
    test_ds  = Subset(full_dataset_eval,  test_indices) if len(test_indices) else None

    train_labels = []
    for idx in train_indices:
        pid = full_dataset_train.pids[idx]
        try:
            train_labels.append(int(full_dataset_train.meta[pid]['pcr']))
        except Exception:
            # fallback (should rarely happen)
            sample = full_dataset_train[idx]
            train_labels.append(int(sample[4]))

    train_labels = np.asarray(train_labels, dtype=np.int64)
    class_counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    class_weights = 1.0 / np.maximum(class_counts, 1.0)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        collate_fn=custom_collate_fn,
        worker_init_fn=lambda wid: worker_init_fn(wid, args.seed),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        collate_fn=custom_collate_fn,
        worker_init_fn=lambda wid: worker_init_fn(wid, args.seed),
    )

    test_loader = None
    if test_ds is not None and len(test_ds) > 0:
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_mem,
            collate_fn=custom_collate_fn,
            worker_init_fn=lambda wid: worker_init_fn(wid, args.seed),
        )
        print(f"[INFO] test_ds size = {len(test_ds)}")
    else:
        print("[INFO] No test set in split json (test_pids empty).")

    # Train-eval dataset for quantiles (train indices, no aug, pid)
    train_eval_ds = Subset(full_dataset_eval, train_indices)
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        collate_fn=custom_collate_fn,
        worker_init_fn=lambda wid: worker_init_fn(wid, args.seed),
    )

    return {
        "full_dataset_train": full_dataset_train,
        "full_dataset_eval": full_dataset_eval,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_eval_loader": train_eval_loader,
        "pin_mem": pin_mem,
    }


# ----------------------------
# Train
# ----------------------------
@torch.no_grad()
def infer_backbone_feat_dim(backbone, device, in_channels=3,
                            d=16, h=64, w=64):
    """
    Infer feature dim of backbone.feature_extractor output.
    Must provide a dummy mask because the feature_extractor requires msks.
    """
    backbone.eval()

    x = torch.zeros(1, in_channels, d, h, w, device=device)
    m = torch.zeros(1, 1, d, h, w, device=device)  # dummy mask channel=1

    # ✅ 关键：不要 backbone(x)；直接走 feature_extractor 并传 mask
    feat = backbone.feature_extractor(x, m)

    # feat could be [B,C,D,H,W] or [B,C]; normalize to [B,C]
    if feat.dim() == 5:
        feat = feat.mean(dim=(2, 3, 4))  # GAP
    feat_dim = int(feat.shape[1])
    return feat_dim


def load_concept_embeddings(concept_pt: str) -> torch.Tensor:
    obj = torch.load(concept_pt, map_location="cpu")
    if isinstance(obj, dict):
        # try common keys
        for k in ["embeddings", "E", "concept_emb", "concept_embeddings"]:
            if k in obj:
                emb = obj[k]
                break
        else:
            # if dict itself is mapping concept->vec
            if all(isinstance(v, torch.Tensor) for v in obj.values()):
                emb = torch.stack(list(obj.values()), dim=0)
            else:
                raise KeyError(f"Cannot find embeddings in dict keys: {list(obj.keys())[:20]}")
    else:
        emb = obj
    if isinstance(emb, np.ndarray):
        emb = torch.from_numpy(emb)
    return emb.float()


def build_backbone_from_ckpt(args, device: str) -> nn.Module:
    model = ResNet3DClassifier(
        in_channels=args.in_channels,
        depth=str(args.resnet_depth),  # '18'/'34'/'50'；你命令行传 50 这里转成字符串最稳
    ).to(device)
    ckpt = torch.load(args.resnet_ckpt, map_location="cpu")
    sd = ckpt.get(args.ckpt_key, ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] missing keys in backbone load: {missing[:5]} ... ({len(missing)})")
    if unexpected:
        print(f"[WARN] unexpected keys in backbone load: {unexpected[:5]} ... ({len(unexpected)})")

    # Freeze backbone
    for p in model.parameters():
        p.requires_grad = False

    model.to(device)
    model.eval()
    return model


def train_main(args):
    set_seed(args.seed)
    device = args.device

    # Load concept embeddings
    concept_emb = load_concept_embeddings(args.concept_pt)

    # Backbone
    backbone = build_backbone_from_ckpt(args, device=device)
    feat_dim = infer_backbone_feat_dim(backbone, device=device, in_channels=args.in_channels)
    print(f"[INFO] inferred feat_dim = {feat_dim}")

    # Data
    pack = build_datasets_and_loaders(args)
    clinical_dim = int(getattr(pack["full_dataset_train"], "clinical_dim", 0)) if getattr(args, "use_clinical", False) else 0
    feature_names = getattr(pack["full_dataset_train"], "clinical_feature_names", None)
    if getattr(args, "use_clinical", False):
        if clinical_dim > 0:
            print(f"[INFO] Clinical fusion enabled, dim = {clinical_dim}")
            if feature_names:
                print(f"[INFO] Clinical features = {feature_names}")
        else:
            print("[WARN] use_clinical=True but no clinical_vec found in json; fallback to CBM image-only.")
    modulation_indices = []
    if getattr(args, "use_clinical_modulation", False):
        modulation_indices = resolve_modulation_indices(feature_names, args.modulation_vars)
        if not modulation_indices:
            raise ValueError("use_clinical_modulation=True but no modulation_vars could be resolved.")
        print(f"[INFO] Clinical modulation enabled, vars = {[feature_names[i] for i in modulation_indices]}")

    model = PCBMRouteA(
        backbone=backbone,
        feat_dim=feat_dim,
        concept_emb=concept_emb,
        use_alpha=args.use_alpha,
        alpha_hidden=args.alpha_hidden,
        clinical_dim=clinical_dim,
        clinical_hidden=args.clinical_hidden,
        use_linear_clinical_head=args.use_linear_clinical_head,
        modulation_input_dim=len(modulation_indices),
        modulation_hidden=args.modulation_hidden,
        modulation_indices=modulation_indices,
    ).to(device)

    train_loader = pack["train_loader"]
    val_loader = pack["val_loader"]
    test_loader = pack["test_loader"]

    # Loss / opt
    criterion = nn.BCEWithLogitsLoss()
    params = list(model.Wf.parameters()) + list(model.head.parameters())
    if model.use_alpha:
        params += list(model.alpha_net.parameters())
    if model.clinical_net is not None:
        params += list(model.clinical_net.parameters())
    if model.modulation_net is not None:
        params += list(model.modulation_net.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    # Warmup + cosine (per-iteration)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_ratio * total_steps)
    min_lr = args.lr * args.min_lr_ratio

    def get_lr(step: int) -> float:
        if not args.use_warmup_cosine:
            return args.lr
        if step < warmup_steps and warmup_steps > 0:
            return args.lr * (step + 1) / warmup_steps
        # cosine
        t = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * t))

    # Logging
    run_name = args.run_name or f"icb_seed{args.seed}_{'alpha' if args.use_alpha else 'noalpha'}"
    out_dir = os.path.join(args.out_root, run_name)
    ensure_dir(out_dir)
    log_csv = os.path.join(out_dir, "train_log.csv")
    f_log, w_log = init_csv_logger(log_csv, ["epoch", "lr", "train_loss", "val_loss", "val_auc", "val_thr",
                                            "test_auc_at_best",
                                            "best_test_f1@0.5","best_test_sens@0.5","best_test_spec@0.5",
                                            "best_test_f1@valThr","best_test_sens@valThr","best_test_spec@valThr","best_test_acc@valThr"])

    best_val_auc = -1.0
    best_val_thr = 0.5
    test_auc_at_best = None
    best_test_m05 = {}
    best_test_mY = {}
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}")
        for batch in pbar:
            if batch is None:
                continue
            # lr
            lr = get_lr(global_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            imgs, msks, labs, clinical, pid = unpack_batch(batch, device, need_pid=False)

            out = model(imgs, msks, clinical)
            logit = out["logit"].view(-1)
            labs  = labs.view(-1).float()  
            loss = criterion(logit, labs)


            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        train_loss = total_loss / max(1, steps)

        # Val
        val_loss, val_auc, val_thr = validate(model, val_loader, criterion, device, split_name="Val")

        # Save per-epoch
        if args.save_all_epochs:
            ckpt_epoch = os.path.join(out_dir, f"epoch{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "val_auc": float(val_auc),
                "val_thr": float(val_thr),
                "args": vars(args),
            }, ckpt_epoch)

        # If best: save best + evaluate test once
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_thr = val_thr

            if test_loader is not None:
                _, test_auc, _, test_metrics = validate(model, test_loader, criterion, device, split_name="Test", fixed_thr=val_thr, compute_youden=False, return_metrics=True)
                test_auc_at_best = float(test_auc)
                # store best-test metrics at fixed threshold (=val_thr)
                best_test_m05 = test_metrics.get("m@0.5", {})
                best_test_mY  = test_metrics.get("m@fixed_thr", {})


            best_path = os.path.join(out_dir, "best.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "best_val_auc": float(best_val_auc),
                "best_val_thr": float(best_val_thr),
                "test_auc_at_best": test_auc_at_best,
                "best_test_metrics": {
                    "@0.5": best_test_m05,
                    "@valThr": best_test_mY,
                    "val_thr": float(best_val_thr),
                },
                "args": vars(args),
            }, best_path)
            print(f"[INFO] New best val AUC {best_val_auc:.4f} at epoch {epoch}, saved {best_path}")

        w_log.writerow([
            epoch, get_lr(global_step-1), train_loss, val_loss, val_auc, val_thr,
            "" if test_auc_at_best is None else test_auc_at_best,
            best_test_m05.get("f1",""), best_test_m05.get("sens",""), best_test_m05.get("spec",""),
            best_test_mY.get("f1",""),  best_test_mY.get("sens",""),  best_test_mY.get("spec",""), best_test_mY.get("acc","")
        ])
        f_log.flush()

        print()  # blank line between epochs


    f_log.close()
    print(f"[DONE] Training finished. Logs in {out_dir}")
    best_path = os.path.join(out_dir, "best.pt")
    return out_dir, best_path


# ----------------------------
# Eval mode main
# ----------------------------
def eval_main(args):
    set_seed(args.seed)
    device = args.device

    if not args.eval_ckpt:
        raise ValueError("--eval_ckpt is required in eval mode")

    ensure_dir(args.out_dir)

    # Load concept embeddings + backbone (needed to construct model)
    concept_emb = load_concept_embeddings(args.concept_pt)
    backbone = build_backbone_from_ckpt(args, device=device)
    feat_dim = infer_backbone_feat_dim(backbone, device=device, in_channels=args.in_channels)

    ckpt = torch.load(args.eval_ckpt, map_location="cpu")
    sd = ckpt.get("model", ckpt)

    # Load stored val_thr if present
    stored_val_thr = ckpt.get("best_val_thr", None)
    if stored_val_thr is not None:
        stored_val_thr = float(stored_val_thr)

    # Data loaders
    pack = build_datasets_and_loaders(args)
    clinical_dim = int(getattr(pack["full_dataset_train"], "clinical_dim", 0)) if getattr(args, "use_clinical", False) else 0
    feature_names = getattr(pack["full_dataset_train"], "clinical_feature_names", None)
    modulation_indices = []
    if getattr(args, "use_clinical_modulation", False):
        modulation_indices = resolve_modulation_indices(feature_names, args.modulation_vars)
        if not modulation_indices:
            raise ValueError("use_clinical_modulation=True but no modulation_vars could be resolved.")
        print(f"[INFO] Clinical modulation enabled, vars = {[feature_names[i] for i in modulation_indices]}")

    model = PCBMRouteA(
        backbone=backbone,
        feat_dim=feat_dim,
        concept_emb=concept_emb,
        use_alpha=args.use_alpha,
        alpha_hidden=args.alpha_hidden,
        clinical_dim=clinical_dim,
        clinical_hidden=args.clinical_hidden,
        use_linear_clinical_head=args.use_linear_clinical_head,
        modulation_input_dim=len(modulation_indices),
        modulation_hidden=args.modulation_hidden,
        modulation_indices=modulation_indices,
    ).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] missing keys in eval ckpt load: {missing[:5]} ... ({len(missing)})")
    if unexpected:
        print(f"[WARN] unexpected keys in eval ckpt load: {unexpected[:5]} ... ({len(unexpected)})")

    train_eval_loader = pack["train_eval_loader"]
    val_loader = pack["val_loader"]
    test_loader = pack["test_loader"]

    # Determine which split to evaluate
    if args.eval_split == "train":
        eval_loader = train_eval_loader
        eval_indices = pack["train_indices"]
        split_name = "train"
    elif args.eval_split == "val":
        eval_loader = val_loader
        eval_indices = pack["val_indices"]
        split_name = "val"
    elif args.eval_split == "test":
        if test_loader is None:
            raise RuntimeError("Split json has no test_pids; cannot eval on test.")
        eval_loader = test_loader
        eval_indices = pack["test_indices"]
        split_name = "test"
    else:
        raise ValueError("--eval_split must be train, val or test")


    # Determine an optional fixed threshold for reporting.
    # By default we only report threshold-free metrics and metrics at thr=0.5.
    val_thr = 0.5
    if args.report_valthr:
        # Load stored val_thr if present
        if stored_val_thr is not None and (not args.force_recompute_val_thr):
            val_thr = stored_val_thr
            print(f"[EVAL] Using val_thr from ckpt: {val_thr:.4f}")
        else:
            # recompute on val split
            criterion = nn.BCEWithLogitsLoss()
            _, _, val_thr = validate(model, val_loader, criterion, device, split_name="Val(for thr)")
            print(f"[EVAL] Recomputed val_thr: {val_thr:.4f}")


    # In this simplified version (no alpha gating), always use raw concept scores c.
    use_c_eff = False
    # (c) Collect baseline concepts/probs on target split
    C0, _, probs0, Y, PIDs, clinical0 = collect_concepts_and_probs(
        model, eval_loader, device, use_c_eff=use_c_eff, need_pid=True
    )
    if len(PIDs) != C0.shape[0]:
        # best effort: if pid list empty, fabricate indices
        if len(PIDs) == 0:
            PIDs = [f"{split_name}_{i:06d}" for i in range(C0.shape[0])]
        else:
            raise RuntimeError(f"PID length mismatch: {len(PIDs)} vs {C0.shape[0]}")
    
    # --- export per-case concept scores & contributions for visualization ---
    np.save(os.path.join(args.out_dir, f"concept_scores_{split_name}_NxK.npy"), C0.astype(np.float32))
    np.save(os.path.join(args.out_dir, f"probs_{split_name}_N.npy"), probs0.astype(np.float32))
    with open(os.path.join(args.out_dir, f"pids_{split_name}.txt"), "w") as f:
        for p in PIDs:
            f.write(str(p) + "\n")

    w = model.head.weight.detach().cpu().numpy().reshape(-1)[:model.K].astype(np.float32)  # [K]
    np.save(os.path.join(args.out_dir, "head_weight_K.npy"), w)
    if getattr(model, "use_clinical", False):
        w_full = model.head.weight.detach().cpu().numpy().reshape(-1).astype(np.float32)
        w_clin = w_full[model.K:]
        np.save(os.path.join(args.out_dir, "head_weight_clinical.npy"), w_clin)
        if getattr(model, "use_linear_clinical_head", False):
            feature_names = getattr(pack["full_dataset_train"], "clinical_feature_names", None)
            if feature_names and len(feature_names) == len(w_clin):
                with open(os.path.join(args.out_dir, "clinical_weight_names.txt"), "w", encoding="utf-8") as f:
                    for n in feature_names:
                        f.write(str(n) + "\n")

    contrib = C0 * w[None, :]
    np.save(os.path.join(args.out_dir, f"concept_contrib_{split_name}_NxK.npy"), contrib.astype(np.float32))

    # Export baseline metrics
    m05 = compute_basic_metrics(Y, probs0, thr=0.5)
    payload = {
        "split": split_name,
        "n": int(len(Y)),
        "metrics@0.5": m05,
    }
    if args.report_valthr:
        my = compute_basic_metrics(Y, probs0, thr=val_thr)
        payload.update({
            "val_thr": float(val_thr),
            "metrics@valThr": my,
        })

    with open(os.path.join(args.out_dir, f"metrics_{split_name}.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # (d) Faithfulness
    if args.do_faithfulness:
        run_faithfulness(
            model=model,
            C0=C0,
            clinical0=clinical0,
            probs0=probs0,
            labels=Y,
            pids=PIDs,
            device=device,
            out_dir=os.path.join(args.out_dir, f"faithfulness_{split_name}"),
            topm_list=args.faith_topm,
            do_random_topm=args.faith_random_topm,
            random_repeats=args.faith_random_repeats,
            seed=args.seed,
            p_change_eps=args.faith_p_change_eps,
        )

    # (e) Reliability (TTA)
    if args.do_reliability:
        # Build TTA dataset with random aug enabled (is_train=True, use_aug=True).
        target_img_size_dce = tuple(args.img_size_dce)
        target_img_size_t2  = tuple(args.img_size_t2) if args.img_size_t2 is not None else target_img_size_dce
        target_img_size_dwi = tuple(args.img_size_dwi) if args.img_size_dwi is not None else target_img_size_dce

        tta_dataset = ISPY2Dataset(
            json_path=args.json_path,
            img_size_dce=target_img_size_dce,
            img_size_t2=target_img_size_t2,
            img_size_dwi=target_img_size_dwi,
            repeat_channels=1,
            return_pid=True,
            is_train=True,
            use_aug=True,
        )
        pin_mem = (args.device.startswith("cuda") and torch.cuda.is_available())

        run_reliability_tta(
            model=model,
            tta_dataset=tta_dataset,
            tta_indices=eval_indices,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_mem=pin_mem,
            base_labels=Y,
            base_pids=PIDs,
            device=device,
            out_dir=os.path.join(args.out_dir, f"reliability_{split_name}"),
            tta_T=args.tta_T,
            seed=args.seed,
            use_c_eff=use_c_eff,
            coverages=args.coverages,
            val_thr=val_thr,
            reliability_rank=args.reliability_rank,
            report_valthr=args.report_valthr,
        )

    print(f"[DONE] Eval finished. Outputs in: {args.out_dir}")

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    # Mode
    ap.add_argument("--mode", type=str, default="train", choices=["train", "eval", "train_eval"])

    # Data / split
    ap.add_argument("--json_path", type=str, default=os.environ.get("ISPY2_JSON_PATH", "./ISPY2_json_minimal.json"))
    ap.add_argument("--split_json", type=str, default=os.environ.get("ISPY2_SPLIT_JSON_PATH", "./split_seed20.json"))
    # Image sizes (D H W). Use the same target size as your preprocessed volumes (e.g., 160 160 160).
    ap.add_argument("--img_size_dce", type=int, nargs=3, default=[160, 160, 160])
    ap.add_argument("--img_size_t2", type=int, nargs=3, default=None)
    ap.add_argument("--img_size_dwi", type=int, nargs=3, default=None)
    ap.add_argument("--strict_split_check", action="store_true", default=False)

    # Backbone
    ap.add_argument("--resnet_depth", type=str, default="50", choices=["18", "34", "50"])
    ap.add_argument("--in_channels", type=int, default=3)
    ap.add_argument("--resnet_ckpt", type=str, required=True, help="Path to your trained resnet best.pt")
    ap.add_argument("--ckpt_key", type=str, default="model", help="state_dict key in checkpoint (default: model)")

    # Concepts
    ap.add_argument("--concept_pt", type=str, required=True, help="Path to concept embeddings .pt, shape [K,Dt]")
    ap.add_argument("--use_alpha", action="store_true", help="Enable alpha gate")
    ap.add_argument("--alpha_hidden", type=int, default=256)
    ap.add_argument("--use_clinical", action="store_true", help="Enable concept + clinical direct fusion")
    ap.add_argument("--clinical_hidden", type=int, default=128)
    ap.add_argument(
        "--use_linear_clinical_head",
        action="store_true",
        help="Use a truly linear fusion head: [concepts, raw_clinical] -> Linear, instead of clinical MLP + linear head",
    )
    ap.add_argument("--use_clinical_modulation", action="store_true", help="Enable residual clinical-guided concept modulation: c' = c * (1 + g(v_sel))")
    ap.add_argument("--modulation_hidden", type=int, default=64)
    ap.add_argument(
        "--modulation_vars",
        type=str,
        nargs="+",
        default=None,
        help="Clinical variables/groups used for modulation. Supports exact feature names plus group tokens: menopause, race",
    )

    # Train hyperparams
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=8)
    # Augmentation: default ON; disable via --no_aug
    ap.add_argument("--no_aug", dest="use_aug", action="store_false", help="Disable augmentation.")
    ap.set_defaults(use_aug=True)
    # Warmup + cosine
    ap.add_argument("--use_warmup_cosine", action="store_true", default=True)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)

    # Output
    ap.add_argument("--out_root", type=str, default="./log_icb")
    ap.add_argument("--run_name", type=str, default="")

    # Checkpoint saving policy
    ap.add_argument("--save_all_epochs", dest="save_all_epochs", action="store_true", help="Save epochXXX.pt for every epoch.")
    ap.add_argument("--no_save_all_epochs", dest="save_all_epochs", action="store_false", help="Do not save per-epoch checkpoints.")
    ap.set_defaults(save_all_epochs=False)
    ap.add_argument("--save_last", action="store_true", default=False, help="Save last.pt at the end.")


    # Eval mode args
    ap.add_argument("--eval_ckpt", type=str, default="", help="Best icb checkpoint path (best.pt)")
    ap.add_argument("--eval_split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--out_dir", type=str, default="./eval_out", help="Eval output directory")
    ap.add_argument("--auto_eval_subdir", type=str, default="eval_auto",
                    help="When --mode train_eval, evaluation outputs go to <out_dir>/<auto_eval_subdir> (unless --out_dir explicitly set).")
    ap.add_argument("--force_recompute_val_thr", action="store_true", default=False)

    ap.add_argument("--do_faithfulness", action="store_true", default=True)
    ap.add_argument("--do_reliability", action="store_true", default=True)

    ap.add_argument("--tta_T", type=int, default=10)
    ap.add_argument("--coverages", type=int, nargs="+", default=[50, 70, 90, 100])

    # Reliability ranking for selective prediction
    ap.add_argument("--reliability_rank", type=str, default="concept_R", choices=["concept_R", "confidence", "prob_std"],
                    help="Score used to rank samples for selective prediction.")
    ap.add_argument("--report_valthr", action="store_true", default=False,
                    help="Also compute/report metrics at val-selected threshold (val_thr). Default: off.")

    ap.add_argument("--faith_topm", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--faith_p_change_eps", type=float, default=0.05,
                    help="Epsilon for p_change_rate: fraction(|Δp| > eps). Default: 0.05")
    ap.add_argument("--faith_random_topm", action="store_true", default=True,
                    help="Also run random-m zeroing baseline for top-m faithfulness")
    ap.add_argument("--faith_random_repeats", type=int, default=5,
                    help="Number of random baselines to average for each m")

    return ap.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        train_main(args)
    elif args.mode == "eval":
        eval_main(args)
    else:
        # train then automatically run eval on the best checkpoint
        out_dir, best_ckpt = train_main(args)
        # If user didn't override --out_dir, put eval outputs under training directory
        if (not args.out_dir) or (args.out_dir == "./eval_out"):
            args.out_dir = os.path.join(out_dir, args.auto_eval_subdir)
        args.eval_ckpt = best_ckpt
        # default split: use args.eval_split
        eval_main(args)


if __name__ == "__main__":
    main()
