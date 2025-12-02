"""
High-level helpers for configuring the Setun–Ternary Clause Machine (STCM).

This module keeps backwards compatibility with the original ``FuzzyPatternTM_STCM``
class while exposing utilities that highlight the newly introduced ternary operator
suite, including the new fuzzy logic t-norms.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

from .operators import (
    available_ternary_operators,
    build_ternary_operator,
    _BaseTernaryOperator,
    # Fuzzy t-norms
    LukasiewiczTNorm,
    GodelTNorm,
    HamacherProduct,
    YagerTNorm,
    DrasticProduct,
    EinsteinProduct,
    NilpotentMinimum,
    BoundedDifference,
    # Learnable operators
    ParameterizedTNorm,
    SoftMinMax,
)
from .tm import FuzzyPatternTM_STCM


# Mapping of operator names to descriptions for documentation
OPERATOR_DESCRIPTIONS: Dict[str, str] = {
    "capacity": "Capacity-mismatch dynamics (default STCM)",
    "product": "Product t-norm with exponential penalty",
    "lukasiewicz": "Bounded product: max(0, a + b - 1) - strong conjunction",
    "godel": "Minimum t-norm: min(a, b) - weak conjunction",
    "hamacher": "Hamacher product: smooth with gradient stability",
    "yager": "Parameterized family: tunable strictness via p parameter",
    "drastic": "Drastic product: very strict, requires near-perfect match",
    "einstein": "Einstein product: alternative smooth product",
    "nilpotent": "Nilpotent minimum: threshold-based activation",
    "bounded_diff": "Bounded difference: emphasizes agreement",
    "parameterized": "Schweizer-Sklar: learnable t-norm parameter",
    "softminmax": "Learnable interpolation between min and product",
    "tqand": "Quantum-inspired ternary AND",
    "txor": "Ternary XOR emphasizing disagreement",
    "tmaj": "Ternary majority vote",
}


class IntegratedSTCM(FuzzyPatternTM_STCM):
    """
    Drop-in STCM variant that exposes helper utilities for the ternary operator
    registry. No behavioural changes are introduced—the base class already
    implements the extended functionality, but this wrapper makes discovery
    easier for downstream projects.
    
    New operators include classical fuzzy logic t-norms (Lukasiewicz, Gödel, 
    Hamacher, Yager) and learnable operators (ParameterizedTNorm, SoftMinMax).
    """

    @staticmethod
    def supported_operators() -> List[str]:
        """Return list of all supported operator names."""
        base = ["capacity", "product"]
        return sorted(set(base + list(available_ternary_operators())))
    
    @staticmethod
    def operator_descriptions() -> Dict[str, str]:
        """Return descriptions of all available operators."""
        return dict(OPERATOR_DESCRIPTIONS)
    
    @staticmethod
    def get_operator_class(name: str) -> Optional[Type[_BaseTernaryOperator]]:
        """Get the operator class for a given name."""
        op = build_ternary_operator(name)
        if op is not None:
            return type(op)
        return None


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

