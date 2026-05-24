import os, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import nibabel as nib
from torch.utils.data import random_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score, roc_curve
from tqdm import tqdm
from monai.transforms import Compose, RandFlipd, RandGaussianNoised, RandAffined
import math
from functools import partial
from typing import Optional
import sys
from torch.amp import GradScaler, autocast
import csv
from datetime import datetime
from torch.utils.data import Subset


from ispy2_dataset_finetune import ISPY2Dataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class CFG:
    USE_AMP = True
    CLIP_BACKBONE = 5.0
    CLIP_HEAD = 2.0
    LOG_GLOBAL_NORM = True

    JSON_PATH = os.environ.get("ISPY2_JSON_PATH", "./ISPY2_json_minimal.json")


    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 20 
    EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-3
    NUM_WORKERS = 8
    ASSERT_SHAPE = None
    USE_AUG = True
    LOG_DIR = os.environ.get("RESNET_LOG_DIR", "./log_resnet_onefold")
    LOG_NAME = "resnet50"                 
    SAVE_BEST = True   
    SPLIT_JSON_PATH = os.environ.get("ISPY2_SPLIT_JSON_PATH", "./split_seed20.json")
    STRICT_SPLIT_CHECK = True 


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    base_seed = CFG.SEED
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)
    torch.manual_seed(base_seed + worker_id)


def custom_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def make_log_paths(cfg):
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    tag = f"{cfg.LOG_NAME}_seed{cfg.SEED}"
    csv_path = os.path.join(cfg.LOG_DIR, f"{tag}.csv")
    txt_path = os.path.join(cfg.LOG_DIR, f"{tag}.txt")
    ckpt_path = os.path.join(cfg.LOG_DIR, f"{tag}_best.pt")
    return csv_path, txt_path, ckpt_path

def load_split_pids(split_json_path: str):
    import json
    with open(split_json_path, "r") as f:
        sp = json.load(f)

    train_pids = sp.get("train_pids", sp.get("train_ids", None))
    val_pids   = sp.get("val_pids",   sp.get("val_ids",   None))
    test_pids  = sp.get("test_pids",  sp.get("test_ids",  None))

    if train_pids is None or val_pids is None:
        raise ValueError(f"[Split] split json 缺少 train/val 字段: {split_json_path}")

    if test_pids is None:
        test_pids = []

    return train_pids, val_pids, test_pids, sp


def pids_to_indices(pids, dataset_pids, strict: bool = True):
    """
    pids: split.json 中的 pid list
    dataset_pids: dataset.pids
    """
    pid2idx = {pid: i for i, pid in enumerate(dataset_pids)}
    indices = []
    missing = []

    for pid in pids:
        if pid in pid2idx:
            indices.append(pid2idx[pid])
        else:
            missing.append(pid)

    if missing and strict:
        raise KeyError(f"[Split] 有 {len(missing)} 个 pid 不在当前 dataset 中，示例: {missing[:10]}")

    if missing and (not strict):
        print(f"[Split][WARN] 丢弃 {len(missing)} 个不在 dataset 的 pid，示例: {missing[:10]}")

    return indices

def export_split_json(
    full_dataset,
    train_indices,
    val_indices,
    test_indices,
    cfg,
    out_dir=None,
):
    """
    根据当前 seed，把 train / val / test 的 pid 划分导出为 json
    """
    import json, os

    if out_dir is None:
        out_dir = cfg.LOG_DIR
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(
        out_dir,
        f"split_seed{cfg.SEED}.json"
    )

    pids = full_dataset.pids

    split_dict = {
        "seed": cfg.SEED,
        "json_path": cfg.JSON_PATH,
        "num_total": len(full_dataset),
        "num_train": len(train_indices),
        "num_val": len(val_indices),
        "num_test": len(test_indices),
        "train_pids": [pids[i] for i in train_indices],
        "val_pids":   [pids[i] for i in val_indices],
        "test_pids":  [pids[i] for i in test_indices],
    }

    with open(out_path, "w") as f:
        json.dump(split_dict, f, indent=2)

    print(f"[Split] exported to: {out_path}")
    return out_path


def init_logger(csv_path, txt_path):
    # CSV 头
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "time", "epoch",
                "train_loss", "train_auc",
                "val_loss", "val_auc", "val_thr",
                "best_val_auc"
            ])

    # TXT 头（可选）
    if not os.path.exists(txt_path):
        with open(txt_path, "w") as f:
            f.write(f"Log created at {datetime.now().isoformat(timespec='seconds')}\n")

def log_epoch(csv_path, txt_path, epoch, train_loss, train_auc, val_loss, val_auc, val_thr, best_val_auc):
    now = datetime.now().isoformat(timespec="seconds")

    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([now, epoch, train_loss, train_auc, val_loss, val_auc, val_thr, best_val_auc])

    with open(txt_path, "a") as f:
        f.write(
            f"[{now}] epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_auc={train_auc:.4f} "
            f"val_loss={val_loss:.4f} val_auc={val_auc:.4f} val_thr={val_thr:.3f} "
            f"best_val_auc={best_val_auc:.4f}\n"
        )

# ===================== 模型定义=====================

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50']


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, dilation=dilation,
                     stride=stride, padding=dilation, bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    if out.size(1) < planes:
        pad = torch.zeros(
            out.size(0), planes - out.size(1),
            out.size(2), out.size(3), out.size(4),
            device=out.device, dtype=out.dtype
        )
        out = torch.cat([out, pad], dim=1)
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        md = x[:, :1]
        x_feat = x[:, 1:]
        residual = x_feat

        out = self.conv1(x_feat); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out);    out = self.bn2(out)

        out = self.ca(out) * out
        if self.downsample is not None:
            residual = self.downsample(x_feat)

        md = md.to(out.dtype)
        md = F.interpolate(md, size=out.shape[2:], mode='trilinear', align_corners=False)
        out = self.sa(md) * out

        out = out + residual
        out = self.relu(out)
        out = torch.cat([md, out], dim=1)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.ca = ChannelAttention(planes * self.expansion)
        self.sa = SpatialAttention()

    def forward(self, x):
        md = x[:, :1]
        x_feat = x[:, 1:]
        residual = x_feat

        out = self.conv1(x_feat); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out);    out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out);    out = self.bn3(out)

        out = self.ca(out) * out
        if self.downsample is not None:
            residual = self.downsample(x_feat)

        md = md.to(out.dtype)
        md = F.interpolate(md, size=out.shape[2:], mode='trilinear', align_corners=False)
        out = self.sa(md) * out

        out = out + residual
        out = self.relu(out)
        out = torch.cat([md, out], dim=1)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, cin=1, shortcut_type='B', no_cuda=False, ndim=64):
        super(ResNet, self).__init__()
        self.no_cuda = no_cuda
        self.inplanes = ndim
        self.shortcut_type = shortcut_type

        self.conv1 = nn.Conv3d(cin, ndim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(ndim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, ndim,   layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, ndim*2, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, ndim*4, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, ndim*8, layers[3], shortcut_type, stride=2)

        self.out_channels = ndim * 8 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block, planes=planes * block.expansion,
                                     stride=stride, no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion),
                )
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride,
                            dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x, md):

        
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)

        md = md.to(x.dtype)
        md = F.interpolate(md, size=x.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([md, x], dim=1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x[:, 1:, ...]
        return x


def resnet18(cin=1, shortcut_type='B', no_cuda=False, ndim=64):
    return ResNet(BasicBlock, [2, 2, 2, 2], cin=cin, shortcut_type=shortcut_type, no_cuda=no_cuda, ndim=ndim)


def resnet34(cin=1, shortcut_type='B', no_cuda=False, ndim=64):
    return ResNet(BasicBlock, [3, 4, 6, 3], cin=cin, shortcut_type=shortcut_type, no_cuda=no_cuda, ndim=ndim)


def resnet50(cin=1, shortcut_type='B', no_cuda=False, ndim=64):
    return ResNet(Bottleneck, [3, 4, 6, 3], cin=cin, shortcut_type=shortcut_type, no_cuda=no_cuda, ndim=ndim)


class ResNet3DClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        depth: str = '18',
        ndim: int = 64,
        shortcut_type: str = 'B',
    ):
        super(ResNet3DClassifier, self).__init__()

        if depth == '18':
            self.feature_extractor = resnet18(cin=in_channels, shortcut_type=shortcut_type, no_cuda=False, ndim=ndim)
        elif depth == '34':
            self.feature_extractor = resnet34(cin=in_channels, shortcut_type=shortcut_type, no_cuda=False, ndim=ndim)
        elif depth == '50':
            self.feature_extractor = resnet50(cin=in_channels, shortcut_type=shortcut_type, no_cuda=False, ndim=ndim)
        else:
            raise ValueError(f"Unsupported depth: {depth}")

        self.gap = nn.AdaptiveAvgPool3d(1)
        feat_dim = getattr(self.feature_extractor, "out_channels", 512)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, imgs, msks=None):
        feat_3d = self.feature_extractor(imgs, msks)
        if feat_3d.ndim == 5:
            feat_3d = self.gap(feat_3d)
            feat_3d = feat_3d.view(feat_3d.size(0), -1)
        logit = self.classifier(feat_3d).squeeze(-1)
        return logit

def named_params_by_group(model: nn.Module):
    head, backbone = [], []
    need_clip = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = n.lower()
        is_norm = ("bn" in lname) or ("norm" in lname)
        if "classifier" in lname or "fc" in lname:
            head.append(p)
            if not is_norm:
                need_clip.append((n, p))
        else:
            backbone.append(p)
            if not is_norm:
                need_clip.append((n, p))
    return head, backbone, need_clip


@torch.no_grad()
def compute_global_grad_norm(named_params):
    total = 0.0
    for _, p in named_params:
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += g.pow(2).sum().item()
    return (total ** 0.5)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    valid_steps = 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="Training", leave=False)
    head, backbone, need_clip = named_params_by_group(model)


    use_amp = (scaler is not None) and getattr(scaler, "is_enabled", lambda: False)() and device.type == "cuda"

    for batch in pbar:
        if batch is None:
            continue

        imgs = batch[6].to(device, non_blocking=True)  # img_aug
        msks = batch[7].to(device, non_blocking=True)  # mask_aug
        labs = batch[4].to(device, non_blocking=True)  # label


        optimizer.zero_grad(set_to_none=True)

        # ======================
        # forward + loss（AMP 只包这部分）
        # ======================
        if use_amp:
            with torch.amp.autocast(device_type="cuda", enabled=True):
                logits = model(imgs, msks)
                loss = criterion(logits, labs)

            # backward（用 scaler）
            scaler.scale(loss).backward()

            # 梯度裁剪前必须先 unscale_
            scaler.unscale_(optimizer)

        else:
            logits = model(imgs, msks)
            loss = criterion(logits, labs)
            loss.backward()

        # ======================
        # grad clip
        # ======================
        if len(head):
            torch.nn.utils.clip_grad_norm_(head, max_norm=CFG.CLIP_HEAD)
        if len(backbone):
            torch.nn.utils.clip_grad_norm_(backbone, max_norm=CFG.CLIP_BACKBONE)

        gnorm = compute_global_grad_norm(need_clip) if CFG.LOG_GLOBAL_NORM else None

        # ======================
        # optimizer step
        # ======================
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # ======================
        # 记录指标
        # ======================
        total_loss += float(loss.item())
        valid_steps += 1

        preds = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labs.detach().cpu().numpy())

        if gnorm is not None:
            pbar.set_postfix(loss=f"{loss.item():.3f}", gnorm=f"{gnorm:.2f}")
        else:
            pbar.set_postfix(loss=f"{loss.item():.3f}")

    avg_loss = total_loss / max(1, valid_steps)
    auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
    return avg_loss, auc



def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    valid_steps = 0
    all_preds, all_labels = [], []

    use_amp = (CFG.USE_AMP and device.type == "cuda")

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)
        for batch in pbar:
            if batch is None:
                continue

            # 解包
            imgs = batch[6].to(device, non_blocking=True)  # img_aug
            msks = batch[7].to(device, non_blocking=True)  # mask_aug
            labs = batch[4].to(device, non_blocking=True)  # label
            if use_amp:
                with torch.amp.autocast(device_type="cuda", enabled=True):
                    logits = model(imgs, msks)
                    loss = criterion(logits, labs)
            else:
                logits = model(imgs, msks)
                loss = criterion(logits, labs)

            total_loss += float(loss.item())
            valid_steps += 1

            preds = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labs.detach().cpu().numpy())

    avg_loss = total_loss / max(1, valid_steps)
    auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5

    probs = np.asarray(all_preds).ravel()
    labels = np.asarray(all_labels).astype(int).ravel()
    preds_bin = (probs >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds_bin, labels=[0, 1]).ravel()
    acc = accuracy_score(labels, preds_bin)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(labels, preds_bin, zero_division=0)

    print(f" Val F1 {f1:.4f} | Acc {acc:.4f} | Sens {sens:.4f} | Spec {spec:.4f} "
          f"| ConfMat [[TN {tn}  FP {fp}] [FN {fn}  TP {tp}]]")

    youden_thr = 0.5
    if len(np.unique(labels)) > 1:
        fpr, tpr, thr = roc_curve(labels, probs)
        youden = tpr - fpr
        yi = int(np.argmax(youden))
        youden_thr = float(thr[yi])
        sens_youden = float(tpr[yi])
        spec_youden = float(1.0 - fpr[yi])

        pred_youden = (probs >= youden_thr).astype(int)
        tn_y, fp_y, fn_y, tp_y = confusion_matrix(labels, pred_youden, labels=[0, 1]).ravel()

        print(f" Val (Youden) Thr {youden_thr:.3f} | Sens {sens_youden:.4f} | Spec {spec_youden:.4f} "
              f"| ConfMat [[TN {tn_y} FP {fp_y}] [FN {fn_y} TP {tp_y}]]")
    else:
        print(" Val (Youden) 跳过：验证集只有单一类别，无法计算 ROC/Youden。")

    return avg_loss, auc, youden_thr


def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
    eps: float = 1e-6,
):
    bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    modulating = (1.0 - p_t).clamp(min=eps).pow(gamma)
    loss = modulating * bce
    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def build_criterion(train_labels: np.ndarray, device):
    p_pos = float((train_labels == 1).mean())
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = max(0.25, min(0.75, 1.0 - p_pos))

    def focal_part(logits, labels):
        return binary_focal_loss_with_logits(
            logits, labels,
            alpha=FOCAL_ALPHA,
            gamma=FOCAL_GAMMA,
            reduction="mean",
        )

    bce = nn.BCEWithLogitsLoss()

    def combined_criterion(logits, labels):
        labels = labels.to(logits.device).float()
        loss_focal = focal_part(logits, labels)
        loss_bce = bce(logits, labels)
        return 0.5 * loss_focal + 0.5 * loss_bce

    return combined_criterion


def main():
    set_seed(CFG.SEED)
    print("当前种子：",CFG.SEED)
    if torch.cuda.is_available():
        print("当前使用物理 GPU 名称：", torch.cuda.get_device_name(torch.cuda.current_device()))
        print("当前逻辑 GPU 索引：", torch.cuda.current_device())
    else:
        print("未检测到 CUDA，将在 CPU 上训练")

    data = json.load(open(CFG.JSON_PATH, "r"))
    all_items = []
   
    for pid, rec in data.items():
        rec = dict(rec)
        rec["id"] = pid
        all_items.append(rec)

    print(f"Total samples: {len(all_items)}")


    device = torch.device(CFG.DEVICE)
    print(f"Using device: {device}")

    # 2. Dataset
    target_img_size_dce = (160, 160, 160)
    target_img_size_t2 = (384, 256, 48)  # 占位
    target_img_size_dwi = (256, 128, 32)  # 占位

    
    # Train Dataset (Augmentation ON)
    full_dataset_train = ISPY2Dataset(
        json_path=CFG.JSON_PATH,
        img_size_dce=target_img_size_dce,
        img_size_t2=target_img_size_t2,
        img_size_dwi=target_img_size_dwi,
        repeat_channels=1,
        return_pid=False,
        is_train=True,
        use_aug=CFG.USE_AUG,
    )
    
    # Eval Dataset (Augmentation OFF)
    full_dataset_eval = ISPY2Dataset(
        json_path=CFG.JSON_PATH,
        img_size_dce=target_img_size_dce,
        img_size_t2=target_img_size_t2,
        img_size_dwi=target_img_size_dwi,
        repeat_channels=1,
        return_pid=False,
        is_train=False, 
        use_aug=False,
    )
    
    # Split

    # ===== Split (from json if provided) =====
    if getattr(CFG, "SPLIT_JSON_PATH", None):
        train_pids, val_pids, test_pids, split_meta = load_split_pids(CFG.SPLIT_JSON_PATH)
        print(f"[Split] Loading split from: {CFG.SPLIT_JSON_PATH}")
        print(f"[Split] train/val/test = {len(train_pids)}/{len(val_pids)}/{len(test_pids)}")

        train_indices = pids_to_indices(train_pids, full_dataset_train.pids, strict=CFG.STRICT_SPLIT_CHECK)
        val_indices   = pids_to_indices(val_pids,   full_dataset_train.pids, strict=CFG.STRICT_SPLIT_CHECK)
        test_indices  = pids_to_indices(test_pids,  full_dataset_train.pids, strict=CFG.STRICT_SPLIT_CHECK)

    else:
 
        N = len(full_dataset_train)
        train_size = int(0.80 * N)
        val_size = N - train_size
        test_size = N - train_size - val_size

        from torch.utils.data import random_split
        dummy_ds = range(N)
        train_subset, val_subset, test_subset = random_split(
            dummy_ds, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(CFG.SEED)
        )
        train_indices = list(train_subset.indices)
        val_indices   = list(val_subset.indices)
        test_indices  = list(test_subset.indices)

    train_ds = Subset(full_dataset_train, train_indices)  # Train: augmentation ON
    val_ds   = Subset(full_dataset_eval,  val_indices)    # Val: augmentation OFF
    test_ds  = Subset(full_dataset_eval,  test_indices)   # Test: augmentation OFF


    train_labels = []
    for idx in train_indices:
        pid = full_dataset_train.pids[idx]
        train_labels.append(int(full_dataset_train.meta[pid]["pcr"]))
    train_labels = np.array(train_labels)

    cls_cnt = np.array([(train_labels == 0).sum(), (train_labels == 1).sum()], dtype=np.float32)
    w_per_cls = np.sqrt(1.0 / (cls_cnt + 1e-6))
    weights = w_per_cls[train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.BATCH_SIZE,
        sampler=sampler,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=pin_mem,
        collate_fn=custom_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=pin_mem,
        collate_fn=custom_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    # test_loader = DataLoader(
    #     test_ds,
    #     batch_size=CFG.BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=CFG.NUM_WORKERS,
    #     pin_memory=pin_mem,
    #     collate_fn=custom_collate_fn,
    #     worker_init_fn=worker_init_fn,
    # )

    model = ResNet3DClassifier(
        in_channels=3, 
        depth='50',
    )
    model = model.to(device)

    criterion = build_criterion(train_labels, device)

    head_params, backbone_params, _ = named_params_by_group(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.LEARNING_RATE,
        weight_decay=CFG.WEIGHT_DECAY
    )
    
    #scaler = None

    scaler = GradScaler("cuda", enabled=CFG.USE_AMP)
    
   # ===== 日志初始化（新增）=====
    csv_path, txt_path, ckpt_path = make_log_paths(CFG)
    init_logger(csv_path, txt_path)

    best_auc = 0.0
    best_test_auc = 0.0

    for epoch in range(CFG.EPOCHS):
        print(f"Epoch {epoch + 1}/{CFG.EPOCHS}")

        avg_loss, train_auc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        print(f"Train Loss: {avg_loss:.4f} | Train AUC: {train_auc:.4f}")

        val_loss, val_auc, val_thr = validate(model, val_loader, criterion, device)

        if val_auc > best_auc:
            best_auc = val_auc
            print(f"New Best Val AUC: ----------------------------{best_auc:.4f}")

            if CFG.SAVE_BEST:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_val_auc": best_auc,
                        "seed": CFG.SEED,
                    },
                    ckpt_path
                )
        else:
            print(f"Val AUC: {val_auc:.4f} (Best: {best_auc:.4f})")

        log_epoch(
            csv_path, txt_path,
            epoch=epoch + 1,
            train_loss=avg_loss,
            train_auc=train_auc,
            val_loss=val_loss,
            val_auc=val_auc,
            val_thr=val_thr,
            best_val_auc=best_auc
        )

    print(f"Training Finished. Best Val AUC: {best_auc:.4f}")


if __name__ == '__main__':
    main()

