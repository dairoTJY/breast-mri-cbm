import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.ndimage as ndimage
import torch
from monai.transforms import Compose, RandAffined, RandFlipd, RandGaussianNoised
from torch.utils.data import Dataset


def pad_or_crop_3d(volume: np.ndarray, target_shape):
    """Center crop or zero-pad a 3D volume to the target shape."""
    assert volume.ndim == 3
    d, h, w = volume.shape
    target_d, target_h, target_w = target_shape

    d_start = max((d - target_d) // 2, 0)
    h_start = max((h - target_h) // 2, 0)
    w_start = max((w - target_w) // 2, 0)

    d_end = d_start + min(target_d, d)
    h_end = h_start + min(target_h, h)
    w_end = w_start + min(target_w, w)

    cropped = volume[d_start:d_end, h_start:h_end, w_start:w_end]

    cropped_d, cropped_h, cropped_w = cropped.shape
    pad_d = max(target_d - cropped_d, 0)
    pad_h = max(target_h - cropped_h, 0)
    pad_w = max(target_w - cropped_w, 0)

    pad_before_d = pad_d // 2
    pad_after_d = pad_d - pad_before_d
    pad_before_h = pad_h // 2
    pad_after_h = pad_h - pad_before_h
    pad_before_w = pad_w // 2
    pad_after_w = pad_w - pad_before_w

    padded = np.pad(
        cropped,
        (
            (pad_before_d, pad_after_d),
            (pad_before_h, pad_after_h),
            (pad_before_w, pad_after_w),
        ),
        mode="constant",
        constant_values=0,
    )
    return padded.astype(np.float32)


def zscore_normalize(volume: np.ndarray, eps: float = 1e-6):
    """Z-score normalize a volume using full-volume statistics."""
    mean = volume.mean()
    std = volume.std()
    return (volume - mean) / (std + eps)


class ISPY2Dataset(Dataset):
    """
    Unified ISPY2 dataset used by both the ResNet and CBM training scripts.

    Returned tuple layout is kept compatible with the training code:
    `index, t2, dwi, dce, label, mask, img_aug, mask_aug, clinical[, pid]`
    """

    def __init__(
        self,
        json_path,
        img_size_dce=(128, 160, 160),
        img_size_t2=None,
        img_size_dwi=None,
        repeat_channels: int = 6,
        return_pid: bool = False,
        is_train: bool = True,
        use_aug: bool = True,
        mask_dilation_iters: int = 3,
        roi_boost_factor: float = 5.0,
    ):
        self.json_path = Path(json_path)
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.pids = list(self.meta.keys())
        self.img_size_dce = tuple(img_size_dce)
        self.img_size_t2 = tuple(img_size_t2) if img_size_t2 is not None else self.img_size_dce
        self.img_size_dwi = tuple(img_size_dwi) if img_size_dwi is not None else self.img_size_dce
        self.repeat_channels = repeat_channels
        self.return_pid = return_pid
        self.mask_dilation_iters = int(mask_dilation_iters)
        self.roi_boost_factor = float(roi_boost_factor)

        self.has_clinical = False
        self.clinical_dim = 0
        self.clinical_feature_names = None
        for pid in self.pids:
            clinical_vec = self.meta[pid].get("clinical_vec", None)
            if isinstance(clinical_vec, (list, tuple)) and len(clinical_vec) > 0:
                self.has_clinical = True
                self.clinical_dim = int(len(clinical_vec))
                names = self.meta[pid].get("clinical_feature_names", None)
                if isinstance(names, list) and len(names) == self.clinical_dim:
                    self.clinical_feature_names = list(names)
                break

        aug_enabled = is_train and use_aug
        self.geom_aug = Compose(
            [
                RandFlipd(keys=["image", "mask"], spatial_axis=2, prob=0.5),
                RandAffined(
                    keys=["image", "mask"],
                    rotate_range=(0.0, 0.0, math.radians(30)),
                    scale_range=(0.1, 0.1, 0.0),
                    mode=("bilinear", "nearest"),
                    padding_mode="border",
                    prob=0.5,
                ),
            ]
        ) if aug_enabled else None

        self.int_aug = Compose(
            [
                RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
            ]
        ) if aug_enabled else None

    def __len__(self):
        return len(self.pids)

    def _load_nii(self, path: str) -> np.ndarray:
        """Load a NIfTI volume as a float32 array with shape [D, H, W]."""
        img = nib.load(str(path))
        data = img.get_fdata().astype(np.float32)

        if data.ndim == 4 and data.shape[-1] == 1:
            data = data[..., 0]
        if data.ndim != 3:
            raise ValueError(f"Expected a 3D volume, got shape {data.shape} for {path}")
        return data

    def _build_clinical_tensor(self, info: dict) -> torch.Tensor:
        if not self.has_clinical:
            return torch.tensor([], dtype=torch.float32)

        clinical_vec = info.get("clinical_vec", None)
        if isinstance(clinical_vec, (list, tuple)) and len(clinical_vec) == self.clinical_dim:
            return torch.tensor(clinical_vec, dtype=torch.float32)
        return torch.zeros((self.clinical_dim,), dtype=torch.float32)

    def __getitem__(self, idx: int):
        pid = self.pids[idx]
        info = self.meta[pid]

        pre_path = info["image_0"]
        post_path = info["image_1"]
        mask_path = info.get("mask")
        label = int(info["pcr"])

        dce_pre = self._load_nii(pre_path)
        dce_post = self._load_nii(post_path)
        mask = self._load_nii(mask_path) if mask_path else np.zeros_like(dce_pre)

        mask_bool = mask > 0
        if self.mask_dilation_iters > 0:
            mask_bool = ndimage.binary_dilation(mask_bool, iterations=self.mask_dilation_iters)
        mask = mask_bool.astype(np.float32)

        dce_sub = dce_post - dce_pre

        pre_norm = zscore_normalize(dce_pre)
        post_norm = zscore_normalize(dce_post)
        sub_norm = zscore_normalize(dce_sub)

        mask_final = pad_or_crop_3d(mask, self.img_size_dce)
        pre_final = pad_or_crop_3d(pre_norm, self.img_size_dce)
        post_final = pad_or_crop_3d(post_norm, self.img_size_dce)
        sub_final = pad_or_crop_3d(sub_norm, self.img_size_dce)

        roi_weight = (mask_final * self.roi_boost_factor + 1.0).astype(np.float32)
        pre_final = pre_final * roi_weight
        post_final = post_final * roi_weight
        sub_final = sub_final * roi_weight

        dce_1ch = sub_final[None, ...]
        if self.repeat_channels > 1:
            dce_multi = np.repeat(dce_1ch, self.repeat_channels, axis=0)
        else:
            dce_multi = dce_1ch

        aug_image = np.concatenate([pre_final[None, ...], post_final[None, ...], dce_1ch], axis=0)

        dce_tensor = torch.from_numpy(dce_multi).float()
        mask_tensor = torch.from_numpy(mask_final[None, ...]).float()

        img_for_aug = torch.tensor(aug_image, dtype=torch.float32)
        mask_for_aug = torch.tensor(mask_final[None, ...], dtype=torch.float32)

        if self.geom_aug is not None:
            aug_data = self.geom_aug({"image": img_for_aug, "mask": mask_for_aug})
            img_aug, mask_aug = aug_data["image"], aug_data["mask"]
        else:
            img_aug, mask_aug = img_for_aug, mask_for_aug

        if self.int_aug is not None:
            img_aug = self.int_aug({"image": img_aug})["image"]

        t2_tensor = torch.zeros((1, *self.img_size_t2), dtype=torch.float32)
        dwi_tensor = torch.zeros((1, *self.img_size_dwi), dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        index_tensor = torch.tensor(idx, dtype=torch.long)
        clinical_tensor = self._build_clinical_tensor(info)

        items = (
            index_tensor,
            t2_tensor,
            dwi_tensor,
            dce_tensor,
            label_tensor,
            mask_tensor,
            img_aug,
            mask_aug,
            clinical_tensor,
        )
        if self.return_pid:
            return items + (pid,)
        return items
