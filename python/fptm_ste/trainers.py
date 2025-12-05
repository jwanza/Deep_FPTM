from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def anneal_ste_factor(module, new_tau: float):
    """
    Recursively set tau (hardening) if present.
    """
    for m in module.modules():
        if hasattr(m, "tau"):
            setattr(m, "tau", float(new_tau))
        if hasattr(m, "set_temperature"):
            m.set_temperature(float(new_tau))
        if hasattr(m, "anneal_temperature"):
            m.anneal_temperature(new_tau)
        if hasattr(m, "anneal_binarizers"):
            m.anneal_binarizers(new_tau)


class TsetlinMarginLoss(nn.Module):
    """
    Differentiable approximation of the Tsetlin Machine margin feedback.
    Loss = ReLU(T - score_correct) + Sum(ReLU(T + score_incorrect))
    """
    def __init__(self, T: float = 1.0):
        super().__init__()
        self.T = T

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        B, C = logits.shape
        # Extract score of correct class
        correct_scores = logits.gather(1, target.view(-1, 1)).squeeze()
        
        # Loss for correct class: max(0, T - score)
        loss_correct = F.relu(self.T - correct_scores)
        
        # Loss for incorrect classes: max(0, T + score)
        loss_incorrect = F.relu(self.T + logits)
        
        # Mask out correct class from incorrect loss
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(1, target.view(-1, 1), False)
        loss_incorrect = loss_incorrect[mask].view(B, C - 1)
        
        return (loss_correct + loss_incorrect.sum(dim=1)).mean()


class EMAWrapper:
    """
    Exponential Moving Average (EMA) helper for model parameters.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
                continue
            shadow_param = self.shadow[name]
            shadow_param.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model: torch.nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def gather_auxiliary_losses(module) -> torch.Tensor:
    """
    Collect auxiliary losses (e.g. entropy regularizers) from submodules.
    """
    total = None
    for m in module.modules():
        if hasattr(m, "attention_entropy_loss"):
            extra = m.attention_entropy_loss()
            if extra is not None:
                total = extra if total is None else total + extra
    if total is None:
        first_param = next(module.parameters(), None)
        device = first_param.device if first_param is not None else torch.device("cpu")
        return torch.tensor(0.0, device=device)
    return total


def update_attention_ema(module):
    """
    Notify submodules that maintain EMA statistics to update after optimizer step.
    """
    for m in module.modules():
        if hasattr(m, "update_attention_ema"):
            m.update_attention_ema()


@dataclass
class TauLiteralScheduleConfig:
    tau_start: float = 0.9
    tau_end: float = 0.5
    literal_start: Optional[float] = None
    literal_end: Optional[float] = None
    warmup_epochs: int = 0
    total_epochs: int = 100
    mode: str = "cosine"  # {"cosine", "linear"}


class TauLiteralScheduler:
    """
    Joint scheduler for STE tau and literal budgets.

    Example:
        cfg = TauLiteralScheduleConfig(tau_start=0.9, tau_end=0.45,
                                       literal_start=8, literal_end=4,
                                       total_epochs=50)
        scheduler = TauLiteralScheduler(cfg)
        for epoch in range(num_epochs):
            scheduler.apply(model, epoch)
    """

    def __init__(self, config: TauLiteralScheduleConfig):
        self.config = config

    def _progress(self, epoch: int) -> float:
        total = max(1, self.config.total_epochs - self.config.warmup_epochs)
        if epoch <= self.config.warmup_epochs:
            return 0.0
        return min(1.0, (epoch - self.config.warmup_epochs) / total)

    def _interpolate(self, start: float, end: float, progress: float) -> float:
        if self.config.mode == "linear":
            return start + (end - start) * progress
        # cosine
        import math
        cos = (1 + math.cos(math.pi * progress)) / 2.0
        return end + (start - end) * cos

    def tau_at(self, epoch: int) -> float:
        prog = self._progress(epoch)
        return self._interpolate(self.config.tau_start, self.config.tau_end, prog)

    def literal_at(self, epoch: int) -> Optional[float]:
        if self.config.literal_start is None or self.config.literal_end is None:
            return None
        prog = self._progress(epoch)
        return self._interpolate(self.config.literal_start, self.config.literal_end, prog)

    def apply(self, module: nn.Module, epoch: int) -> None:
        tau_val = self.tau_at(epoch)
        anneal_ste_factor(module, tau_val)
        literal_val = self.literal_at(epoch)
        if literal_val is None:
            return
        for sub in module.modules():
            if hasattr(sub, "literal_budget"):
                sub.literal_budget = float(literal_val)
            if hasattr(sub, "lf"):
                sub.lf = float(literal_val)


def train_step(
    model,
    data,
    target,
    optimizer,
    use_ste: bool = True,
    clip_grad: float = 1.0,
    margin_loss: bool = False,
    l1_lambda: float = 0.0,
    *,
    label_smoothing: float = 0.0,
    teacher_logits: torch.Tensor | None = None,
    distill_weight: float = 0.0,
):
    model.train()
    optimizer.zero_grad()
    out = model(data, use_ste=use_ste)
    if isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out

    if margin_loss:
        T = getattr(model, 'T', 1.0)
        if hasattr(model, 'module') and hasattr(model.module, 'T'):
            T = model.module.T
        loss_fn = TsetlinMarginLoss(T=T)
        ce_loss = loss_fn(logits, target)
    else:
        if target.ndim > 1:
            log_probs = F.log_softmax(logits, dim=-1)
            ce_loss = -(target * log_probs).sum(dim=-1).mean()
        else:
            ce_loss = F.cross_entropy(logits, target, label_smoothing=label_smoothing)

    loss = ce_loss
    if distill_weight > 0.0 and teacher_logits is not None:
        teacher_probs = F.softmax(teacher_logits.detach(), dim=-1)
        student_log_probs = F.log_softmax(logits, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        loss = (1.0 - distill_weight) * loss + distill_weight * kl_loss

    if l1_lambda > 0.0:
        l1_reg = torch.tensor(0.0, device=logits.device)
        for name, param in model.named_parameters():
            if 'ta_' in name or ('logits' in name and 'vote' not in name):
                l1_reg += torch.abs(param).sum()
        loss = loss + l1_lambda * l1_reg

    aux_loss = gather_auxiliary_losses(model)
    if aux_loss is not None:
        loss = loss + aux_loss
    loss.backward()
    if clip_grad and clip_grad > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    optimizer.step()
    update_attention_ema(model)
    return loss.item()




def cosine_anneal_temperature(initial: float, final: float, epoch: int, max_epochs: int) -> float:
    """
    Cosine schedule from initial -> final over [0, max_epochs].
    """
    import math
    if max_epochs <= 0:
        return final
    cos = (1 + math.cos(math.pi * epoch / max_epochs)) / 2.0
    return final + (initial - final) * cos


# =============================================================================
# Curriculum Learning for Clauses
# =============================================================================


class ClauseCurriculumScheduler:
    """
    Curriculum learning scheduler for TM clause training.
    
    Implements progressive training strategies:
    1. LF annealing: Start with high lf (easy matching), gradually reduce
    2. Ternary band annealing: Start with large band, anneal to target
    3. Clause activation: Start with few clauses, progressively add more
    4. Temperature annealing: Soft to hard STE quantization
    
    Args:
        model: TM model to schedule
        total_epochs: Total training epochs
        lf_schedule: (start, end) for lf annealing
        band_schedule: (start, end) for ternary_band annealing
        temp_schedule: (start, end) for temperature annealing
        clause_warmup_epochs: Epochs to warmup clause activation
        schedule_type: 'linear', 'cosine', or 'exponential'
    """
    
    def __init__(
        self,
        model: nn.Module,
        total_epochs: int,
        lf_schedule: tuple = None,  # (start_lf, end_lf)
        band_schedule: tuple = None,  # (start_band, end_band)
        temp_schedule: tuple = None,  # (start_temp, end_temp)
        clause_warmup_epochs: int = 0,
        schedule_type: str = "cosine",
    ):
        self.model = model
        self.total_epochs = total_epochs
        self.lf_schedule = lf_schedule
        self.band_schedule = band_schedule
        self.temp_schedule = temp_schedule
        self.clause_warmup_epochs = clause_warmup_epochs
        self.schedule_type = schedule_type
        self.current_epoch = 0
        
        # Store initial values
        self._initial_lf = self._get_attr("lf")
        self._initial_band = self._get_attr("ternary_band")
        self._initial_temp = self._get_attr("ste_temperature")
    
    def _get_attr(self, name: str):
        """Get attribute from model or first submodule that has it."""
        for m in self.model.modules():
            if hasattr(m, name):
                return getattr(m, name)
        return None
    
    def _set_attr(self, name: str, value):
        """Set attribute on all submodules that have it."""
        for m in self.model.modules():
            if hasattr(m, name):
                setattr(m, name, value)
    
    def _interpolate(self, start: float, end: float, progress: float) -> float:
        """Interpolate between start and end based on schedule type."""
        import math
        
        if self.schedule_type == "linear":
            return start + (end - start) * progress
        elif self.schedule_type == "cosine":
            cos = (1 + math.cos(math.pi * progress)) / 2.0
            return end + (start - end) * cos
        elif self.schedule_type == "exponential":
            # Exponential decay from start to end
            decay = math.log(end / start) if start > 0 else 0
            return start * math.exp(decay * progress)
        else:
            return end  # Default to final value
    
    def step(self, epoch: int = None):
        """Update curriculum for the current epoch."""
        if epoch is not None:
            self.current_epoch = epoch
        
        progress = min(1.0, self.current_epoch / max(1, self.total_epochs))
        
        # LF annealing
        if self.lf_schedule is not None:
            start_lf, end_lf = self.lf_schedule
            new_lf = int(round(self._interpolate(start_lf, end_lf, progress)))
            self._set_attr("lf", new_lf)
        
        # Ternary band annealing
        if self.band_schedule is not None:
            start_band, end_band = self.band_schedule
            new_band = self._interpolate(start_band, end_band, progress)
            self._set_attr("ternary_band", new_band)
        
        # Temperature annealing
        if self.temp_schedule is not None:
            start_temp, end_temp = self.temp_schedule
            new_temp = self._interpolate(start_temp, end_temp, progress)
            self._set_attr("ste_temperature", new_temp)
            anneal_ste_factor(self.model, new_temp)
        
        # Clause warmup (progressive activation)
        if self.clause_warmup_epochs > 0 and self.current_epoch < self.clause_warmup_epochs:
            warmup_progress = self.current_epoch / self.clause_warmup_epochs
            self._apply_clause_warmup(warmup_progress)
        
        self.current_epoch += 1
    
    def _apply_clause_warmup(self, progress: float):
        """Apply progressive clause activation."""
        for m in self.model.modules():
            if hasattr(m, "clause_dropout"):
                # High dropout initially, reduce over warmup
                dropout = 1.0 - progress
                m.clause_dropout = dropout * 0.5  # Max 50% dropout
    
    def get_current_values(self) -> dict:
        """Get current curriculum values for logging."""
        return {
            "lf": self._get_attr("lf"),
            "ternary_band": self._get_attr("ternary_band"),
            "temperature": self._get_attr("ste_temperature"),
            "epoch": self.current_epoch,
            "progress": self.current_epoch / max(1, self.total_epochs),
        }


class AdaptiveLRScheduler:
    """
    Learning rate scheduler that adapts based on clause statistics.
    
    Monitors clause activity and adjusts learning rate to prevent
    collapse or exploding gradients.
    """
    
    def __init__(
        self,
        optimizer,
        base_lr: float,
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
        patience: int = 5,
        factor: float = 0.5,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.patience = patience
        self.factor = factor
        
        self.best_loss = float('inf')
        self.wait = 0
        self.current_lr = base_lr
    
    def step(self, loss: float, clause_variance: float = None):
        """
        Update learning rate based on loss and clause statistics.
        
        Args:
            loss: Current training loss
            clause_variance: Variance of clause outputs (if available)
        """
        # Check for improvement
        if loss < self.best_loss:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
        
        # Reduce LR on plateau
        if self.wait >= self.patience:
            new_lr = max(self.min_lr, self.current_lr * self.factor)
            if new_lr != self.current_lr:
                self.current_lr = new_lr
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
            self.wait = 0
        
        # Adjust based on clause variance if provided
        if clause_variance is not None:
            if clause_variance < 0.01:  # Clauses collapsed
                # Increase LR to encourage differentiation
                new_lr = min(self.max_lr, self.current_lr * 1.1)
                self.current_lr = new_lr
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr


# =============================================================================
# Contrastive Clause Learning
# =============================================================================


class ClauseContrastiveLoss(nn.Module):
    """
    Contrastive loss for clause representations.
    
    Encourages clauses to learn discriminative patterns by maximizing
    similarity between clause outputs of same-class samples and
    minimizing similarity between different-class samples.
    
    Based on NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    
    Args:
        temperature: Temperature scaling for softmax
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, temperature: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        clause_outputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss on clause representations.
        
        Args:
            clause_outputs: [batch, n_clauses] clause strength outputs
            labels: [batch] class labels
            
        Returns:
            Scalar contrastive loss
        """
        batch_size = clause_outputs.shape[0]
        device = clause_outputs.device
        
        # Normalize clause outputs
        clause_norm = F.normalize(clause_outputs, dim=1, p=2)
        
        # Compute similarity matrix
        similarity = torch.mm(clause_norm, clause_norm.t()) / self.temperature
        
        # Create positive/negative masks based on labels
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        positive_mask = labels_equal.float()
        # Exclude self-similarity
        positive_mask.fill_diagonal_(0)
        
        # Negative mask (different class)
        negative_mask = (~labels_equal).float()
        
        # For each sample, compute InfoNCE-style loss
        # log(exp(pos_sim) / sum(exp(all_sim)))
        
        # Numerical stability: subtract max
        sim_max = similarity.max(dim=1, keepdim=True).values
        similarity_stable = similarity - sim_max
        
        # Compute log_sum_exp of all similarities
        exp_sim = torch.exp(similarity_stable)
        log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        
        # Positive term: mean of positive similarities
        positive_sim = similarity_stable * positive_mask
        n_positives = positive_mask.sum(dim=1, keepdim=True).clamp(min=1)
        positive_mean = positive_sim.sum(dim=1, keepdim=True) / n_positives
        
        # Loss: -positive_mean + log_sum_exp
        loss_per_sample = -positive_mean.squeeze() + log_sum_exp.squeeze()
        
        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        return loss_per_sample


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss for TM clause learning.
    
    Extends SupCon loss to work with clause outputs, encouraging
    intra-class compactness and inter-class separation.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        contrast_mode: str = "all",  # 'one' or 'all'
    ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: [batch, dim] normalized feature vectors (clause outputs)
            labels: [batch] class labels
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute anchor-contrast similarity
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Create mask for positive pairs (same class)
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)
        
        # Mask out self-contrast
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positive pairs
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(
            mask_pos_pairs < 1e-6,
            torch.ones_like(mask_pos_pairs),
            mask_pos_pairs
        )
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class ClauseRepresentationLoss(nn.Module):
    """
    Combined loss for learning good clause representations.
    
    Combines:
    1. Classification loss (cross-entropy)
    2. Contrastive loss (clause discrimination)
    3. Diversity loss (encourage diverse clause patterns)
    """
    
    def __init__(
        self,
        contrastive_weight: float = 0.1,
        diversity_weight: float = 0.01,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.diversity_weight = diversity_weight
        self.contrastive_loss = ClauseContrastiveLoss(temperature=temperature)
    
    def forward(
        self,
        logits: torch.Tensor,
        clause_outputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple:
        """
        Compute combined loss.
        
        Returns:
            (total_loss, loss_dict with individual components)
        """
        # Classification loss
        ce_loss = F.cross_entropy(logits, labels)
        
        # Contrastive loss
        contrastive = self.contrastive_loss(clause_outputs, labels)
        
        # Diversity loss: encourage clauses to be different from each other
        # High similarity between clauses = low diversity
        clause_norm = F.normalize(clause_outputs, dim=0)  # normalize per clause
        clause_sim = torch.mm(clause_norm.t(), clause_norm)  # [n_clauses, n_clauses]
        # Penalize high off-diagonal similarity
        diversity_loss = (clause_sim.triu(diagonal=1) ** 2).mean()
        
        # Total loss
        total = ce_loss + self.contrastive_weight * contrastive + self.diversity_weight * diversity_loss
        
        return total, {
            "ce_loss": ce_loss.item(),
            "contrastive_loss": contrastive.item(),
            "diversity_loss": diversity_loss.item(),
            "total_loss": total.item(),
        }


# =============================================================================
# Training Utilities
# =============================================================================


def train_epoch_with_curriculum(
    model: nn.Module,
    dataloader,
    optimizer,
    curriculum: ClauseCurriculumScheduler,
    use_contrastive: bool = False,
    contrastive_weight: float = 0.1,
    device: torch.device = None,
    clip_grad: float = 1.0,
) -> dict:
    """
    Train for one epoch with curriculum learning.
    
    Returns:
        Dictionary with training statistics
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    contrastive_loss_fn = ClauseContrastiveLoss() if use_contrastive else None
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        if isinstance(output, tuple):
            logits, clause_outputs = output
        else:
            logits = output
            clause_outputs = None
        
        # Compute loss
        loss = F.cross_entropy(logits, target)
        
        # Add contrastive loss if enabled
        if use_contrastive and clause_outputs is not None:
            contrastive = contrastive_loss_fn(clause_outputs, target)
            loss = loss + contrastive_weight * contrastive
        
        # Auxiliary losses
        aux_loss = gather_auxiliary_losses(model)
        if aux_loss is not None:
            loss = loss + aux_loss
        
        # Backward pass
        loss.backward()
        
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        # Statistics
        total_loss += loss.item() * target.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == target).sum().item()
        total_samples += target.size(0)
    
    # Update curriculum
    curriculum.step()
    
    extra_logs = {}
    if hasattr(model, "get_gate_diagnostics"):
        extra_logs["gate"] = model.get_gate_diagnostics()
    if hasattr(model, "last_attention_weights"):
        attn = getattr(model, "last_attention_weights")
        if attn is not None:
            extra_logs["attention_mean"] = float(attn.mean().item())
    result = {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "curriculum": curriculum.get_current_values(),
    }
    result.update(extra_logs)
    return result


