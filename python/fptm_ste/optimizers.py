"""
SOTA Optimizers for TM training.
Implements: SAM, LAMB, EMA
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Callable, Iterable, Optional
from collections import defaultdict


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization.
    Finds flatter minima for better generalization.
    
    Usage:
        optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=1e-3)
        for x, y in loader:
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.first_step(zero_grad=True)
            criterion(model(x), y).backward()
            optimizer.second_step(zero_grad=True)
    """
    
    def __init__(self, params, base_optimizer: type, rho: float = 0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
    
    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """Ascend to find the worst point in neighborhood."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]['e_w'] = e_w
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """Descend from the perturbed point."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]['e_w'])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def step(self, closure=None):
        raise NotImplementedError("Use first_step() and second_step()")


class EMA(nn.Module):
    """
    Exponential Moving Average of model parameters.
    
    Usage:
        ema = EMA(model, decay=0.999)
        # In training loop:
        ema.update()
        # For evaluation:
        ema.apply_shadow()
        evaluate(model)
        ema.restore()
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        super().__init__()
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float = 0.0):
    """Cosine schedule with linear warmup."""
    import math
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class LAMB(torch.optim.Optimizer):
    """LAMB optimizer for large batch training."""
    
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-6, weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('LAMB does not support sparse gradients')
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decoupled weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                # Adam step
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                exp_avg_corrected = exp_avg / bias_correction1
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2
                
                adam_step = exp_avg_corrected / (exp_avg_sq_corrected.sqrt() + group['eps'])
                
                # LAMB trust ratio
                weight_norm = p.norm(2)
                adam_norm = adam_step.norm(2)
                
                if weight_norm > 0 and adam_norm > 0:
                    trust_ratio = weight_norm / adam_norm
                else:
                    trust_ratio = 1.0
                
                p.add_(adam_step, alpha=-group['lr'] * trust_ratio)
        
        return loss
