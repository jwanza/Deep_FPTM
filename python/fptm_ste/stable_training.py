"""
Stable Training Utilities for STCM/DeepTM

This module provides training utilities that achieve stable, non-sporadic learning
by implementing key principles from the Julia FuzzyPatternTM:

1. Momentum-based clause weight updates with EMA tracking
2. Adaptive learning rate based on clause activity
3. Warmup/cooldown schedules for exploration vs exploitation
4. Sample-reweighting based on confidence
5. Gradient accumulation with probabilistic gating

These utilities work with both IncrementalSTCM and standard FuzzyPatternTM_STCM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class StableTrainingConfig:
    """Configuration for stable training regime."""
    
    # EMA configuration
    use_ema: bool = True
    ema_decay: float = 0.995  # Higher = more stable
    ema_warmup_epochs: int = 5
    
    # Learning rate adaptation
    base_lr: float = 0.001
    min_lr: float = 1e-6
    max_lr: float = 0.01
    lr_warmup_epochs: int = 3
    lr_decay_epochs: int = 50
    lr_decay_factor: float = 0.1
    
    # Clause activity monitoring
    target_clause_activity: float = 0.3  # Target fraction of active clauses
    activity_adaptation_rate: float = 0.1
    
    # Gradient stabilization
    clip_grad: float = 1.0
    gradient_accumulation_steps: int = 1
    use_gradient_checkpointing: bool = False
    
    # Sample weighting
    use_confidence_weighting: bool = True
    confidence_temperature: float = 2.0
    min_sample_weight: float = 0.1
    
    # Probabilistic update gating (Julia-style)
    use_probabilistic_updates: bool = True
    base_update_probability: float = 0.8
    confidence_modulated_updates: bool = True
    
    # Regularization
    clause_l1_weight: float = 0.0001  # Encourage sparse clauses
    vote_l2_weight: float = 0.0001
    diversity_weight: float = 0.001  # Encourage diverse clauses
    
    # Checkpointing
    checkpoint_best_n: int = 5
    checkpoint_metric: str = "val_accuracy"


class StableEMA:
    """
    Enhanced EMA wrapper with warmup and adaptive decay.
    
    Features:
    - Warmup period where decay starts low and increases
    - Adaptive decay based on loss trajectory
    - Separate tracking for different parameter groups
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.995,
        warmup_steps: int = 1000,
    ):
        self.model = model
        self.base_decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
        # Shadow parameters
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadows
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()
        
        # Loss history for adaptive decay
        self._loss_history: List[float] = []
    
    def _get_decay(self) -> float:
        """Get current decay factor with warmup."""
        if self.step_count < self.warmup_steps:
            # Linear warmup
            progress = self.step_count / self.warmup_steps
            return self.base_decay * progress + 0.9 * (1 - progress)
        return self.base_decay
    
    def update(self, loss: Optional[float] = None) -> None:
        """Update shadow parameters with current model weights."""
        decay = self._get_decay()
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in self.shadow:
                    self.shadow[name] = param.detach().clone()
                    continue
                
                # EMA update
                self.shadow[name].mul_(decay).add_(param.detach(), alpha=1 - decay)
        
        self.step_count += 1
        
        if loss is not None:
            self._loss_history.append(loss)
    
    def apply_shadow(self) -> None:
        """Apply shadow weights to model (for evaluation)."""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self) -> None:
        """Restore original weights from backup."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self) -> Dict:
        """Get state dict for checkpointing."""
        return {
            'shadow': {k: v.cpu() for k, v in self.shadow.items()},
            'step_count': self.step_count,
            'loss_history': self._loss_history[-100:],  # Keep last 100
        }
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load state dict."""
        device = next(self.model.parameters()).device
        self.shadow = {k: v.to(device) for k, v in state_dict['shadow'].items()}
        self.step_count = state_dict['step_count']
        self._loss_history = state_dict.get('loss_history', [])


class AdaptiveLRScheduler:
    """
    Learning rate scheduler that adapts based on:
    1. Warmup phase
    2. Clause activity
    3. Loss trajectory
    
    Keeps learning rate stable when training is progressing,
    reduces when stalled, increases when under-training.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: StableTrainingConfig,
    ):
        self.optimizer = optimizer
        self.config = config
        self.current_lr = config.base_lr
        self.epoch = 0
        self.step = 0
        
        # History for adaptation
        self._loss_history: List[float] = []
        self._activity_history: List[float] = []
        self._patience_counter = 0
    
    def _warmup_factor(self) -> float:
        """Compute warmup factor (0 to 1)."""
        if self.epoch >= self.config.lr_warmup_epochs:
            return 1.0
        return (self.epoch + 1) / self.config.lr_warmup_epochs
    
    def _decay_factor(self) -> float:
        """Compute decay factor based on epoch."""
        if self.epoch <= self.config.lr_warmup_epochs:
            return 1.0
        
        decay_progress = (self.epoch - self.config.lr_warmup_epochs) / self.config.lr_decay_epochs
        return math.pow(self.config.lr_decay_factor, decay_progress)
    
    def step_epoch(self, val_loss: Optional[float] = None, clause_activity: Optional[float] = None) -> float:
        """
        Update learning rate at epoch boundary.
        
        Args:
            val_loss: Validation loss (for adaptive adjustment)
            clause_activity: Average clause activity (for adaptive adjustment)
            
        Returns:
            New learning rate
        """
        self.epoch += 1
        
        # Base LR with warmup and decay
        base = self.config.base_lr * self._warmup_factor() * self._decay_factor()
        
        # Adapt based on clause activity
        if clause_activity is not None:
            self._activity_history.append(clause_activity)
            
            # If activity is too low, increase LR to encourage learning
            if clause_activity < self.config.target_clause_activity * 0.5:
                base *= 1.2
            # If activity is too high (all clauses firing), decrease LR
            elif clause_activity > self.config.target_clause_activity * 2:
                base *= 0.8
        
        # Adapt based on loss trajectory
        if val_loss is not None:
            self._loss_history.append(val_loss)
            
            if len(self._loss_history) >= 3:
                recent_avg = sum(self._loss_history[-3:]) / 3
                older_avg = sum(self._loss_history[-6:-3]) / 3 if len(self._loss_history) >= 6 else recent_avg
                
                # Reduce LR if loss stalled
                if recent_avg >= older_avg * 0.99:
                    self._patience_counter += 1
                    if self._patience_counter >= 3:
                        base *= 0.5
                        self._patience_counter = 0
                else:
                    self._patience_counter = 0
        
        # Clamp to bounds
        self.current_lr = max(self.config.min_lr, min(self.config.max_lr, base))
        
        # Apply to optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
        
        return self.current_lr
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr


class ConfidenceWeightedLoss(nn.Module):
    """
    Loss function that weights samples by confidence.
    
    High-confidence samples (easy ones) get lower weight.
    Low-confidence samples (hard/uncertain ones) get higher weight.
    
    This prevents the model from over-fitting to easy samples
    and encourages learning from challenging examples.
    """
    
    def __init__(
        self,
        temperature: float = 2.0,
        min_weight: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.min_weight = min_weight
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute confidence-weighted cross-entropy loss.
        
        Args:
            logits: [batch, n_classes] model outputs
            targets: [batch] ground truth labels
            return_weights: Whether to return sample weights
            
        Returns:
            Loss tensor (and optionally weights)
        """
        # Compute per-sample cross-entropy
        ce = F.cross_entropy(logits, targets, reduction='none')
        
        # Compute confidence (probability of predicted class)
        probs = F.softmax(logits / self.temperature, dim=-1)
        pred_probs = probs.max(dim=-1).values
        
        # Weight: lower confidence = higher weight
        # Using (1 - prob) creates anti-correlation
        weights = 1.0 - pred_probs.detach()
        weights = weights.clamp(min=self.min_weight)
        
        # Normalize weights
        weights = weights / weights.mean()
        
        # Weighted loss
        weighted_loss = ce * weights
        
        if self.reduction == "mean":
            loss = weighted_loss.mean()
        elif self.reduction == "sum":
            loss = weighted_loss.sum()
        else:
            loss = weighted_loss
        
        if return_weights:
            return loss, weights
        return loss


class ClauseRegularizer(nn.Module):
    """
    Regularization terms specific to Tsetlin Machine clauses.
    
    Includes:
    1. L1 on clause weights (sparsity)
    2. L2 on voting weights (stability)
    3. Diversity loss (encourage different clause patterns)
    """
    
    def __init__(
        self,
        clause_l1: float = 0.0001,
        vote_l2: float = 0.0001,
        diversity: float = 0.001,
    ):
        super().__init__()
        self.clause_l1 = clause_l1
        self.vote_l2 = vote_l2
        self.diversity = diversity
    
    def forward(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Compute regularization terms.
        
        Args:
            model: Model to regularize
            
        Returns:
            Dict with individual regularization terms
        """
        device = next(model.parameters()).device
        
        terms = {
            'clause_l1': torch.tensor(0.0, device=device),
            'vote_l2': torch.tensor(0.0, device=device),
            'diversity': torch.tensor(0.0, device=device),
        }
        
        for name, param in model.named_parameters():
            # Clause weights (literal logits)
            if 'logits' in name and 'vote' not in name:
                if self.clause_l1 > 0:
                    terms['clause_l1'] = terms['clause_l1'] + torch.abs(param).mean()
            
            # Voting weights
            if 'voting' in name or 'vote' in name:
                if self.vote_l2 > 0:
                    terms['vote_l2'] = terms['vote_l2'] + (param ** 2).mean()
        
        # Diversity: encourage different clause patterns
        if self.diversity > 0:
            diversity_loss = self._compute_diversity_loss(model)
            terms['diversity'] = diversity_loss
        
        # Scale by weights
        terms['clause_l1'] = terms['clause_l1'] * self.clause_l1
        terms['vote_l2'] = terms['vote_l2'] * self.vote_l2
        terms['diversity'] = terms['diversity'] * self.diversity
        
        return terms
    
    def _compute_diversity_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute diversity loss to encourage different clause patterns."""
        device = next(model.parameters()).device
        diversity = torch.tensor(0.0, device=device)
        
        count = 0
        for name, param in model.named_parameters():
            if ('pos_logits' in name or 'neg_logits' in name) and param.dim() == 2:
                # param is [n_clauses, n_features]
                # Compute pairwise cosine similarity
                normalized = F.normalize(param, dim=1)
                similarity = torch.mm(normalized, normalized.t())
                
                # Penalize high off-diagonal similarity
                n = similarity.size(0)
                mask = 1.0 - torch.eye(n, device=device)
                off_diag = similarity * mask
                
                diversity = diversity + (off_diag ** 2).mean()
                count += 1
        
        if count > 0:
            diversity = diversity / count
        
        return diversity


def stable_train_step(
    model: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    config: StableTrainingConfig,
    loss_fn: Optional[nn.Module] = None,
    regularizer: Optional[ClauseRegularizer] = None,
    accumulation_step: int = 0,
    update_probability: float = 1.0,
) -> Dict[str, float]:
    """
    Single stable training step with all enhancements.
    
    Args:
        model: Model to train
        data: Input batch
        target: Target labels
        optimizer: Optimizer
        config: Training configuration
        loss_fn: Loss function (default: confidence-weighted CE)
        regularizer: Clause regularizer
        accumulation_step: Current gradient accumulation step
        update_probability: Probability of actually updating (Julia-style)
        
    Returns:
        Dict with step statistics
    """
    model.train()
    stats = {}
    
    # Probabilistic update gating (Julia-style)
    if config.use_probabilistic_updates and update_probability < 1.0:
        if torch.rand(1).item() > update_probability:
            # Skip this update
            with torch.no_grad():
                output = model(data)
                logits = output[0] if isinstance(output, tuple) else output
                loss = F.cross_entropy(logits, target)
                stats['loss'] = loss.item()
                stats['skipped'] = True
            return stats
    
    # Only zero gradients at start of accumulation
    if accumulation_step == 0:
        optimizer.zero_grad()
    
    # Forward pass
    output = model(data)
    if isinstance(output, tuple):
        logits, clause_outputs = output
    else:
        logits = output
        clause_outputs = None
    
    # Compute loss
    if loss_fn is not None:
        if config.use_confidence_weighting:
            loss, weights = loss_fn(logits, target, return_weights=True)
            stats['avg_weight'] = weights.mean().item()
        else:
            loss = loss_fn(logits, target)
    else:
        loss = F.cross_entropy(logits, target)
    
    # Add regularization
    if regularizer is not None:
        reg_terms = regularizer(model)
        for name, term in reg_terms.items():
            loss = loss + term
            stats[f'reg_{name}'] = term.item()
    
    # Scale loss for gradient accumulation
    if config.gradient_accumulation_steps > 1:
        loss = loss / config.gradient_accumulation_steps
    
    # Backward pass
    loss.backward()
    
    # Only step optimizer after accumulation complete
    if (accumulation_step + 1) % config.gradient_accumulation_steps == 0:
        # Gradient clipping
        if config.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        
        optimizer.step()
        optimizer.zero_grad()
        stats['optimizer_stepped'] = True
    
    # Compute metrics
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        stats['accuracy'] = (preds == target).float().mean().item()
        stats['loss'] = loss.item() * config.gradient_accumulation_steps
        
        if clause_outputs is not None:
            stats['clause_activity'] = (clause_outputs.abs() > 0.1).float().mean().item()
    
    stats['skipped'] = False
    return stats


def stable_train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: StableTrainingConfig,
    ema: Optional[StableEMA] = None,
    lr_scheduler: Optional[AdaptiveLRScheduler] = None,
    device: torch.device = None,
    epoch: int = 0,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Train for one epoch with all stability enhancements.
    
    Args:
        model: Model to train
        dataloader: Training data
        optimizer: Optimizer
        config: Training configuration
        ema: EMA wrapper (optional)
        lr_scheduler: LR scheduler (optional)
        device: Device
        epoch: Current epoch number
        verbose: Print progress
        
    Returns:
        Epoch statistics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.train()
    
    # Setup loss and regularizer
    if config.use_confidence_weighting:
        loss_fn = ConfidenceWeightedLoss(
            temperature=config.confidence_temperature,
            min_weight=config.min_sample_weight,
        )
    else:
        loss_fn = None
    
    regularizer = ClauseRegularizer(
        clause_l1=config.clause_l1_weight,
        vote_l2=config.vote_l2_weight,
        diversity=config.diversity_weight,
    )
    
    # Compute update probability (decreases with epoch for stability)
    if config.confidence_modulated_updates:
        # Start high, decay over epochs
        update_prob = config.base_update_probability * math.pow(0.99, epoch)
    else:
        update_prob = 1.0
    
    # Accumulate statistics
    total_loss = 0.0
    total_acc = 0.0
    total_activity = 0.0
    total_samples = 0
    total_skipped = 0
    
    accumulation_step = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        stats = stable_train_step(
            model, data, target, optimizer, config,
            loss_fn=loss_fn,
            regularizer=regularizer,
            accumulation_step=accumulation_step,
            update_probability=update_prob,
        )
        
        batch_size = target.size(0)
        total_loss += stats['loss'] * batch_size
        total_acc += stats['accuracy'] * batch_size
        total_samples += batch_size
        
        if stats.get('skipped', False):
            total_skipped += 1
        
        if 'clause_activity' in stats:
            total_activity += stats['clause_activity'] * batch_size
        
        # Update accumulation step
        accumulation_step = (accumulation_step + 1) % config.gradient_accumulation_steps
        
        # Update EMA
        if ema is not None and stats.get('optimizer_stepped', False):
            ema.update(stats['loss'])
        
        if verbose and batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}: loss={stats['loss']:.4f}, acc={stats['accuracy']:.4f}")
    
    # Compute epoch averages
    epoch_stats = {
        'loss': total_loss / max(1, total_samples),
        'accuracy': total_acc / max(1, total_samples),
        'clause_activity': total_activity / max(1, total_samples),
        'skipped_batches': total_skipped,
        'update_probability': update_prob,
    }
    
    return epoch_stats


@torch.no_grad()
def stable_evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    ema: Optional[StableEMA] = None,
    device: torch.device = None,
) -> Dict[str, float]:
    """
    Evaluate model, optionally using EMA weights.
    
    Args:
        model: Model to evaluate
        dataloader: Evaluation data
        ema: EMA wrapper (will use shadow weights if provided)
        device: Device
        
    Returns:
        Evaluation statistics
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Apply EMA weights if available
    if ema is not None:
        ema.apply_shadow()
    
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        logits = output[0] if isinstance(output, tuple) else output
        
        loss = F.cross_entropy(logits, target, reduction='sum')
        total_loss += loss.item()
        
        preds = logits.argmax(dim=1)
        total_correct += (preds == target).sum().item()
        total_samples += target.size(0)
    
    # Restore original weights
    if ema is not None:
        ema.restore()
    
    return {
        'loss': total_loss / max(1, total_samples),
        'accuracy': total_correct / max(1, total_samples),
    }


class StableTrainer:
    """
    High-level trainer with all stability features.
    
    Combines:
    - EMA weight tracking
    - Adaptive learning rate
    - Confidence-weighted loss
    - Clause regularization
    - Probabilistic updates
    - Best model checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[StableTrainingConfig] = None,
        device: torch.device = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config or StableTrainingConfig()
        self.device = device or next(model.parameters()).device
        
        # Setup EMA
        if self.config.use_ema:
            warmup_steps = self.config.ema_warmup_epochs * 100  # Approximate
            self.ema = StableEMA(model, decay=self.config.ema_decay, warmup_steps=warmup_steps)
        else:
            self.ema = None
        
        # Setup LR scheduler
        self.lr_scheduler = AdaptiveLRScheduler(optimizer, self.config)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
        }
        
        # Best model tracking
        self._best_models: List[Tuple[float, Dict]] = []
        self.current_epoch = 0
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.current_epoch += 1
        
        # Training
        train_stats = stable_train_epoch(
            self.model, train_loader, self.optimizer, self.config,
            ema=self.ema,
            lr_scheduler=self.lr_scheduler,
            device=self.device,
            epoch=self.current_epoch,
            verbose=verbose,
        )
        
        self.history['train_loss'].append(train_stats['loss'])
        self.history['train_accuracy'].append(train_stats['accuracy'])
        
        # Validation
        if val_loader is not None:
            val_stats = stable_evaluate(self.model, val_loader, ema=self.ema, device=self.device)
            self.history['val_loss'].append(val_stats['loss'])
            self.history['val_accuracy'].append(val_stats['accuracy'])
            
            # Update LR scheduler
            new_lr = self.lr_scheduler.step_epoch(
                val_loss=val_stats['loss'],
                clause_activity=train_stats.get('clause_activity'),
            )
            self.history['learning_rate'].append(new_lr)
            
            # Track best models
            self._update_best_models(val_stats['accuracy'])
            
            if verbose:
                print(f"Epoch {self.current_epoch}: "
                      f"train_loss={train_stats['loss']:.4f}, "
                      f"train_acc={train_stats['accuracy']:.4f}, "
                      f"val_loss={val_stats['loss']:.4f}, "
                      f"val_acc={val_stats['accuracy']:.4f}, "
                      f"lr={new_lr:.6f}")
            
            return {**train_stats, **{f'val_{k}': v for k, v in val_stats.items()}}
        else:
            self.lr_scheduler.step_epoch()
            
            if verbose:
                print(f"Epoch {self.current_epoch}: "
                      f"train_loss={train_stats['loss']:.4f}, "
                      f"train_acc={train_stats['accuracy']:.4f}")
            
            return train_stats
    
    def _update_best_models(self, metric: float) -> None:
        """Track best N models."""
        state = {
            'model_state_dict': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
            'epoch': self.current_epoch,
            'metric': metric,
        }
        
        if self.ema is not None:
            state['ema_state_dict'] = self.ema.state_dict()
        
        self._best_models.append((metric, state))
        self._best_models.sort(key=lambda x: x[0], reverse=True)
        self._best_models = self._best_models[:self.config.checkpoint_best_n]
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Full training loop."""
        
        for epoch in range(epochs):
            self.train_epoch(train_loader, val_loader, verbose=verbose)
        
        return self.history
    
    def get_best_model(self) -> Optional[Dict]:
        """Get best model checkpoint."""
        if self._best_models:
            return self._best_models[0][1]
        return None
    
    def load_best_model(self) -> None:
        """Load best model weights."""
        best = self.get_best_model()
        if best is not None:
            self.model.load_state_dict(best['model_state_dict'])
            if self.ema is not None and 'ema_state_dict' in best:
                self.ema.load_state_dict(best['ema_state_dict'])






