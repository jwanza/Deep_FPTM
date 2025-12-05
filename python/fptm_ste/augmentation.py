"""
SOTA Data Augmentation for TM training.
Implements: RandAugment, CutMix, MixUp, AutoAugment
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from torchvision import transforms


def cutmix(images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """CutMix augmentation."""
    batch_size = images.size(0)
    indices = torch.randperm(batch_size, device=images.device)
    
    lam = np.random.beta(alpha, alpha)
    
    H, W = images.shape[2], images.shape[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    images_mixed = images.clone()
    images_mixed[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return images_mixed, labels, labels[indices], lam


def mixup(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """MixUp augmentation."""
    batch_size = images.size(0)
    indices = torch.randperm(batch_size, device=images.device)
    lam = np.random.beta(alpha, alpha)
    images_mixed = lam * images + (1 - lam) * images[indices]
    return images_mixed, labels, labels[indices], lam


def mixup_cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """Loss for mixup/cutmix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class CutMixMixUp(nn.Module):
    """Apply CutMix or MixUp randomly."""
    def __init__(self, cutmix_alpha: float = 1.0, mixup_alpha: float = 0.8, cutmix_prob: float = 0.5):
        super().__init__()
        self.cutmix_alpha = cutmix_alpha
        self.mixup_alpha = mixup_alpha
        self.cutmix_prob = cutmix_prob
    
    def forward(self, images: torch.Tensor, labels: torch.Tensor):
        if np.random.random() < self.cutmix_prob:
            return cutmix(images, labels, self.cutmix_alpha)
        return mixup(images, labels, self.mixup_alpha)


def get_cifar10_train_transform(mode: str = 'strong'):
    """Get CIFAR-10 transform with modern augmentation."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    if mode == 'basic':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    
    elif mode == 'strong':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=14),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    
    elif mode == 'auto':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    
    elif mode == 'trivial':
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_cifar10_test_transform():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class RandomErasing(nn.Module):
    """Random Erasing augmentation (on tensors)."""
    def __init__(self, p: float = 0.5, scale: Tuple[float, float] = (0.02, 0.33), ratio: Tuple[float, float] = (0.3, 3.3)):
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if np.random.random() > self.p:
            return img
        
        _, H, W = img.shape
        area = H * W
        
        for _ in range(10):
            target_area = np.random.uniform(*self.scale) * area
            aspect_ratio = np.random.uniform(*self.ratio)
            
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if w < W and h < H:
                x1 = np.random.randint(0, W - w)
                y1 = np.random.randint(0, H - h)
                img[:, y1:y1+h, x1:x1+w] = torch.randn_like(img[:, y1:y1+h, x1:x1+w])
                return img
        return img


