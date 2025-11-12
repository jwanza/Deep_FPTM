from __future__ import annotations

from typing import Iterable

import torch

from .tm import FuzzyPatternTM_STE
from .deep_tm import DeepTMNetwork


def _iter_tm_modules(module) -> Iterable[FuzzyPatternTM_STE]:
    if isinstance(module, FuzzyPatternTM_STE):
        yield module
    elif isinstance(module, DeepTMNetwork):
        for layer in module.layers:
            yield from _iter_tm_modules(layer)
        yield from _iter_tm_modules(module.classifier)
    else:
        for child in getattr(module, "children", lambda: [])():
            yield from _iter_tm_modules(child)


def _apply_symmetric_prior(tm: FuzzyPatternTM_STE) -> None:
    with torch.no_grad():
        tm.ta_pos.zero_()
        tm.ta_neg.zero_()
        tm.ta_pos_inv.zero_()
        tm.ta_neg_inv.zero_()
        half = tm.ta_pos.shape[0]
        features = tm.ta_pos.shape[1]
        if half > 0 and features > 0:
            indices = torch.arange(features, device=tm.ta_pos.device)
            mask = (indices % 2 == 0).float()
            tm.ta_pos += (mask - 0.5) * 0.1
            tm.ta_neg -= (mask - 0.5) * 0.1
        tm.clause_bias.data.fill_(0.05)


def apply_tm_prior_template(module, template: str | None) -> None:
    if template is None or template.lower() in {"", "none"}:
        return
    template = template.lower()
    for tm in _iter_tm_modules(module):
        if template == "symmetric":
            _apply_symmetric_prior(tm)
        elif template == "zero":
            with torch.no_grad():
                tm.ta_pos.zero_()
                tm.ta_neg.zero_()
                tm.ta_pos_inv.zero_()
                tm.ta_neg_inv.zero_()
                tm.clause_bias.data.zero_()
        else:
            raise ValueError(f"Unknown TM prior template '{template}'.")

