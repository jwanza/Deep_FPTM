"""
High-level helpers for configuring the Setun–Ternary Clause Machine (STCM).

This module keeps backwards compatibility with the original ``FuzzyPatternTM_STCM``
class while exposing utilities that highlight the newly introduced ternary operator
suite.
"""

from __future__ import annotations

from typing import Any, List

from .operators import available_ternary_operators
from .tm import FuzzyPatternTM_STCM


class IntegratedSTCM(FuzzyPatternTM_STCM):
    """
    Drop-in STCM variant that exposes helper utilities for the ternary operator
    registry. No behavioural changes are introduced—the base class already
    implements the extended functionality, but this wrapper makes discovery
    easier for downstream projects.
    """

    @staticmethod
    def supported_operators() -> List[str]:
        base = ["capacity", "product"]
        return sorted(set(base + list(available_ternary_operators())))


def build_stcm(operator: str = "capacity", **kwargs: Any) -> FuzzyPatternTM_STCM:
    """
    Convenience constructor that validates the operator argument and returns an
    STCM instance configured with the requested behaviour.
    """

    operator = operator.lower()
    if operator not in IntegratedSTCM.supported_operators():
        raise ValueError(
            f"Unsupported operator '{operator}'. Expected one of {IntegratedSTCM.supported_operators()}."
        )
    return IntegratedSTCM(operator=operator, **kwargs)

