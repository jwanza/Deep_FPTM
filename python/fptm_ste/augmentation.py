"""
Advanced Data Augmentation for Tsetlin Machines.

Implements modern augmentation techniques that mix samples and labels,
improving generalization and regularization.

Key Techniques:
1. Mixup - Linear interpolation of samples and labels
2. CutMix - Cut and paste patches with mixed labels
3. ManifoldMixup - Mixup in hidden layers
4. CutOut - Randomly mask image patches

Benefits:
- Smoother decision boundaries
- Better calibration
- Reduced overfitting
- Works with any architecture

References:
- Zhang et al. (2017): mixup: Beyond Empirical Risk Minimization
- Yun et al. (2019): CutMix: Regularization Strategy to Train Strong Classifiers
- Verma et al. (2019): Manifold Mixup: Better Representations by Interpolating Hidden States
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sample_lambda(alpha: float) -> float:
    """Sample mixing coefficient from Beta distribution."""
    if alpha > 0:
        return np.random.beta(alpha, alpha)
    return 1.0


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply Mixup augmentation to a batch.
    
    Mixup creates virtual training examples by linearly interpolating
    pairs of samples and their labels.
    
    Args:
        x: Input features [batch_size, ...]
        y: Labels [batch_size]
        alpha: Beta distribution parameter (higher = more mixing)
        device: Device for tensors
        
    Returns:
        Tuple of:
        - mixed_x: Mixed input features
        - y_a: Original labels
        - y_b: Shuffled labels
        - lam: Mixing coefficient
    
    Example:
        >>> mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.4)
        >>> pred = model(mixed_x)
        >>> loss = lam * criterion(pred, y_a) + (1-lam) * criterion(pred, y_b)
    """
    if device is None:
        device = x.device
    
    batch_size = x.size(0)
    
    # Sample mixing coefficient
    lam = _sample_lambda(alpha)
    
    # Random permutation for pairing
    index = torch.randperm(batch_size, device=device)
    
    # Mix inputs
    mixed_x = lam * x + (1 - lam) * x[index]
    
    # Return both sets of labels
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: Callable,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Compute mixup loss as weighted combination of two losses.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: Mixing coefficient
        
    Returns:
        Weighted loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def _rand_bbox(
    size: Tuple[int, ...],
    lam: float,
) -> Tuple[int, int, int, int]:
    """
    Generate random bounding box for CutMix.
    
    Args:
        size: Input size (B, C, H, W) or (B, H, W)
        lam: Mixing coefficient
        
    Returns:
        Box coordinates (x1, y1, x2, y2)
    """
    if len(size) == 4:
        W, H = size[3], size[2]
    elif len(size) == 3:
        W, H = size[2], size[1]
    else:
        # For 2D feature vectors, use square grid
        side = int(size[1] ** 0.5)
        if side * side != size[1]:
            # Cannot do cutmix on non-square features
            return 0, 0, 0, 0
        W, H = side, side
    
    # Box size proportional to (1-lam)
    cut_ratio = np.sqrt(1 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    
    # Random center
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Box coordinates (clamped)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    
    return x1, y1, x2, y2


def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply CutMix augmentation to a batch.
    
    CutMix cuts a patch from one image and pastes it onto another,
    with labels mixed proportionally to the patch area.
    
    Args:
        x: Input features [batch_size, channels, height, width] or [batch_size, features]
        y: Labels [batch_size]
        alpha: Beta distribution parameter
        device: Device for tensors
        
    Returns:
        Tuple of:
        - mixed_x: CutMix'd input features
        - y_a: Original labels
        - y_b: Shuffled labels
        - lam: Actual mixing coefficient (adjusted for box area)
    """
    if device is None:
        device = x.device
    
    batch_size = x.size(0)
    
    # Sample mixing coefficient
    lam = _sample_lambda(alpha)
    
    # Random permutation
    index = torch.randperm(batch_size, device=device)
    
    # Get bounding box
    x1, y1, x2, y2 = _rand_bbox(x.size(), lam)
    
    # Handle different input shapes
    if len(x.size()) == 4:
        # Image data: [B, C, H, W]
        mixed_x = x.clone()
        mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((x2 - x1) * (y2 - y1) / (x.size(2) * x.size(3)))
        
    elif len(x.size()) == 3:
        # [B, H, W]
        mixed_x = x.clone()
        mixed_x[:, y1:y2, x1:x2] = x[index, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (x.size(1) * x.size(2)))
        
    elif len(x.size()) == 2:
        # Feature vector: treat as flattened square
        side = int(x.size(1) ** 0.5)
        if side * side == x.size(1):
            # Reshape, apply, reshape back
            x_sq = x.view(batch_size, 1, side, side)
            mixed_sq = x_sq.clone()
            mixed_sq[:, :, y1:y2, x1:x2] = x_sq[index, :, y1:y2, x1:x2]
            mixed_x = mixed_sq.view(batch_size, -1)
            lam = 1 - ((x2 - x1) * (y2 - y1) / (side * side))
        else:
            # Cannot apply cutmix to non-square features, fall back to mixup
            mixed_x = lam * x + (1 - lam) * x[index]
    else:
        # Default to mixup behavior
        mixed_x = lam * x + (1 - lam) * x[index]
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutout(
    x: torch.Tensor,
    n_holes: int = 1,
    hole_size: int = 16,
) -> torch.Tensor:
    """
    Apply Cutout augmentation (random masking).
    
    Args:
        x: Input [batch_size, channels, height, width] or [batch_size, features]
        n_holes: Number of patches to cut out
        hole_size: Size of each square patch
        
    Returns:
        Input with patches masked to zero
    """
    batch_size = x.size(0)
    result = x.clone()
    
    if len(x.size()) == 4:
        h, w = x.size(2), x.size(3)
        
        for _ in range(n_holes):
            y_c = np.random.randint(h, size=batch_size)
            x_c = np.random.randint(w, size=batch_size)
            
            for i in range(batch_size):
                y1 = max(0, y_c[i] - hole_size // 2)
                y2 = min(h, y_c[i] + hole_size // 2)
                x1 = max(0, x_c[i] - hole_size // 2)
                x2 = min(w, x_c[i] + hole_size // 2)
                result[i, :, y1:y2, x1:x2] = 0
                
    elif len(x.size()) == 2:
        # Feature vector
        n_features = x.size(1)
        mask_size = min(hole_size, n_features // 4)
        
        for _ in range(n_holes):
            starts = np.random.randint(n_features - mask_size, size=batch_size)
            for i in range(batch_size):
                result[i, starts[i]:starts[i] + mask_size] = 0
    
    return result


class ManifoldMixup(nn.Module):
    """
    Manifold Mixup - Apply mixup at intermediate layers.
    
    This module wraps a model and applies mixup at a randomly
    selected hidden layer during training.
    
    Args:
        model: Model to wrap (must have named forward hooks)
        alpha: Beta distribution parameter
        mixup_layers: List of layer names where mixup can be applied
    
    Example:
        >>> base_model = MyModel()
        >>> model = ManifoldMixup(base_model, alpha=0.4, mixup_layers=['layer1', 'layer2'])
        >>> pred, y_a, y_b, lam = model(x, y)
    """
    
    def __init__(
        self,
        model: nn.Module,
        alpha: float = 1.0,
        mixup_layers: Optional[list] = None,
    ):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.mixup_layers = mixup_layers or []
        
        # State for mixup
        self._lam: float = 1.0
        self._index: Optional[torch.Tensor] = None
        self._mixup_layer: Optional[str] = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward hooks on specified layers."""
        self._hooks = []
        
        for name, module in self.model.named_modules():
            if name in self.mixup_layers:
                hook = module.register_forward_hook(self._mixup_hook)
                self._hooks.append((name, hook))
    
    def _mixup_hook(self, module: nn.Module, input: tuple, output: torch.Tensor) -> torch.Tensor:
        """Hook function to apply mixup at a layer."""
        # Get the name of this module
        for name, mod in self.model.named_modules():
            if mod is module and name == self._mixup_layer:
                # Apply mixup
                if self._index is not None:
                    return self._lam * output + (1 - self._lam) * output[self._index]
        return output
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Forward pass with manifold mixup.
        
        Args:
            x: Input features
            y: Labels
            training: Whether in training mode
            
        Returns:
            Tuple of (predictions, y_a, y_b, lambda)
        """
        if training and self.mixup_layers:
            batch_size = x.size(0)
            
            # Sample mixup coefficient
            self._lam = _sample_lambda(self.alpha)
            
            # Random permutation
            self._index = torch.randperm(batch_size, device=x.device)
            
            # Choose random layer
            self._mixup_layer = np.random.choice(self.mixup_layers)
            
            # Forward pass (hooks will apply mixup)
            pred = self.model(x)
            
            y_a, y_b = y, y[self._index]
            
            return pred, y_a, y_b, self._lam
        else:
            pred = self.model(x)
            return pred, y, y, 1.0
    
    def __del__(self):
        """Clean up hooks."""
        if hasattr(self, '_hooks'):
            for name, hook in self._hooks:
                hook.remove()


class AugmentationPipeline(nn.Module):
    """
    Unified augmentation pipeline for training.
    
    Combines multiple augmentation strategies with configurable probabilities.
    
    Args:
        use_mixup: Apply mixup
        use_cutmix: Apply cutmix
        use_cutout: Apply cutout
        mixup_alpha: Alpha for mixup/cutmix
        cutout_holes: Number of cutout patches
        cutout_size: Size of cutout patches
        mixup_prob: Probability of using mixup
        cutmix_prob: Probability of using cutmix
        cutout_prob: Probability of using cutout
    """
    
    def __init__(
        self,
        use_mixup: bool = True,
        use_cutmix: bool = True,
        use_cutout: bool = False,
        mixup_alpha: float = 1.0,
        cutout_holes: int = 1,
        cutout_size: int = 16,
        mixup_prob: float = 0.5,
        cutmix_prob: float = 0.5,
        cutout_prob: float = 0.5,
    ):
        super().__init__()
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.use_cutout = use_cutout
        self.mixup_alpha = mixup_alpha
        self.cutout_holes = cutout_holes
        self.cutout_size = cutout_size
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.cutout_prob = cutout_prob
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply augmentation pipeline.
        
        Args:
            x: Input features
            y: Labels
            
        Returns:
            Tuple of (augmented_x, y_a, y_b, lambda)
        """
        lam = 1.0
        y_a, y_b = y, y
        
        if not self.training:
            return x, y_a, y_b, lam
        
        # Choose augmentation
        r = np.random.rand()
        
        if self.use_mixup and r < self.mixup_prob:
            x, y_a, y_b, lam = mixup_data(x, y, self.mixup_alpha)
            
        elif self.use_cutmix and r < self.mixup_prob + self.cutmix_prob:
            x, y_a, y_b, lam = cutmix_data(x, y, self.mixup_alpha)
        
        # Cutout can be applied on top
        if self.use_cutout and np.random.rand() < self.cutout_prob:
            x = cutout(x, self.cutout_holes, self.cutout_size)
        
        return x, y_a, y_b, lam
    
    def compute_loss(
        self,
        criterion: Callable,
        pred: torch.Tensor,
        y_a: torch.Tensor,
        y_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """
        Compute augmentation-aware loss.
        
        Args:
            criterion: Loss function
            pred: Model predictions
            y_a: First label set
            y_b: Second label set
            lam: Mixing coefficient
            
        Returns:
            Weighted loss
        """
        return mixup_criterion(criterion, pred, y_a, y_b, lam)


class FeatureMixup(nn.Module):
    """
    Mixup specifically designed for TM clause features.
    
    Applies mixup at the clause level rather than input level,
    which can be beneficial for Tsetlin Machines.
    
    Args:
        alpha: Beta distribution parameter
    """
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(
        self,
        clause_outputs: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup to clause outputs.
        
        Args:
            clause_outputs: Clause activations [batch_size, n_clauses]
            y: Labels [batch_size]
            
        Returns:
            Tuple of (mixed_clauses, y_a, y_b, lambda)
        """
        if not self.training:
            return clause_outputs, y, y, 1.0
        
        return mixup_data(clause_outputs, y, self.alpha)


class LabelSmoothing(nn.Module):
    """
    Label smoothing loss for better calibration.
    
    Distributes some probability mass from the true class to other classes.
    
    Args:
        n_classes: Number of classes
        smoothing: Smoothing factor (0 = no smoothing, 1 = uniform)
    """
    
    def __init__(self, n_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.n_classes = n_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute label-smoothed cross entropy loss.
        
        Args:
            pred: Predictions [batch_size, n_classes]
            target: Labels [batch_size]
            
        Returns:
            Smoothed loss
        """
        log_probs = F.log_softmax(pred, dim=-1)
        
        # Create smoothed targets
        smooth_targets = torch.full_like(log_probs, self.smoothing / self.n_classes)
        smooth_targets.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # Compute loss
        loss = (-smooth_targets * log_probs).sum(dim=-1).mean()
        
        return loss


class AugmentedTrainer:
    """
    Training wrapper with built-in augmentation.
    
    Simplifies training loop with automatic augmentation handling.
    
    Args:
        model: Model to train
        criterion: Loss function
        optimizer: Optimizer
        augmentation: Augmentation pipeline
        label_smoothing: Label smoothing factor
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        augmentation: Optional[AugmentationPipeline] = None,
        label_smoothing: float = 0.0,
    ):
        self.model = model
        self.base_criterion = criterion
        self.optimizer = optimizer
        self.augmentation = augmentation or AugmentationPipeline()
        
        if label_smoothing > 0:
            # Determine n_classes from model if possible
            n_classes = getattr(model, 'n_classes', 10)
            self.criterion = LabelSmoothing(n_classes, label_smoothing)
        else:
            self.criterion = criterion
    
    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single training step with augmentation.
        
        Args:
            x: Input features
            y: Labels
            
        Returns:
            Tuple of (loss, predictions)
        """
        self.model.train()
        self.augmentation.train()
        
        # Apply augmentation
        aug_x, y_a, y_b, lam = self.augmentation(x, y)
        
        # Forward pass
        pred = self.model(aug_x)
        if isinstance(pred, tuple):
            pred = pred[0]  # Handle TM returning (logits, clauses)
        
        # Compute loss
        loss = self.augmentation.compute_loss(self.criterion, pred, y_a, y_b, lam)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss, pred
    
    def eval_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single evaluation step (no augmentation).
        
        Args:
            x: Input features
            y: Labels
            
        Returns:
            Tuple of (loss, predictions)
        """
        self.model.eval()
        
        with torch.no_grad():
            pred = self.model(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            loss = self.base_criterion(pred, y)
        
        return loss, pred

