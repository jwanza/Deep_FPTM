import torch
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


def train_step(model, data, target, optimizer, use_ste: bool = True, clip_grad: float = 1.0):
    model.train()
    optimizer.zero_grad()
    out = model(data, use_ste=use_ste)
    if isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out
    loss = F.cross_entropy(logits, target)
    loss.backward()
    if clip_grad and clip_grad > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    optimizer.step()
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


