from copy import deepcopy

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


def train_step(
    model, 
    data, 
    target, 
    optimizer, 
    use_ste: bool = True, 
    clip_grad: float = 1.0, 
    margin_loss: bool = False, 
    l1_lambda: float = 0.0
):
    model.train()
    optimizer.zero_grad()
    out = model(data, use_ste=use_ste)
    if isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out
    
    if margin_loss:
        # Attempt to get T from model
        T = getattr(model, 'T', 1.0)
        if hasattr(model, 'module') and hasattr(model.module, 'T'):
            T = model.module.T
        loss_fn = TsetlinMarginLoss(T=T)
        loss = loss_fn(logits, target)
    else:
        loss = F.cross_entropy(logits, target)

    # L1 Regularization
    if l1_lambda > 0.0:
        l1_reg = torch.tensor(0.0, device=logits.device)
        for name, param in model.named_parameters():
            # Heuristic to find literal inclusion logits
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


