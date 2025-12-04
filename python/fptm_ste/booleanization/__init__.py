"""
Booleanization Solutions for Tsetlin Machines.

This package implements advanced methods to overcome the fundamental
booleanization bottleneck in Tsetlin Machines, where continuous features
are converted to binary and information is lost.

Modules:
1. continuous_residual - Dual-stream binary + continuous architecture
2. probabilistic - Distributional literals with uncertainty
3. hyperdimensional - HD computing for similarity-preserving encoding
4. information_bottleneck - Optimal compression binarization
5. hierarchical - Multi-resolution clause hierarchy
6. attention_adaptive - Per-sample dynamic binarization

Each module provides a TM-compatible model that addresses the booleanization
problem from a different perspective, and they can be combined in the
ultimate_hybrid architecture.
"""

from .continuous_residual import (
    ContinuousResidualClauseMachine,
    DualStreamTM,
    SoftThresholdBinarizer,
)

from .probabilistic import (
    ProbabilisticLiteralClauseMachine,
    DistributionalLiteral,
    UncertaintyAwareVoting,
)

from .hyperdimensional import (
    HyperdimensionalClauseMachine,
    HDEncoder,
    LevelHVEncoder,
)

from .information_bottleneck import (
    InformationBottleneckBinarizer,
    InformationPreservingClauseMachine,
    VIBLayer,
)

from .hierarchical import (
    HierarchicalMultiResolutionTM,
    HierarchicalLevel,
    CrossLevelAttention,
)

from .attention_adaptive import (
    NeuralSymbolicTransformer,
    NeuralSymbolicBlock,
    DynamicThresholdPredictor,
)


from .learnable import (
    LearnableBinarizer,
)
from .enhanced_continuous import (
    EnhancedContinuousTM,
    EnhancedContinuousEncoder,
    MultiScaleThermometer,
    GaussianBasisExpansion,
    LearnedFeatureBins,
    PositionalValueEncoding,
)

__all__ = [
    # Continuous Residual
    "ContinuousResidualClauseMachine",
    "DualStreamTM",
    "SoftThresholdBinarizer",
    # Probabilistic
    "ProbabilisticLiteralClauseMachine",
    "DistributionalLiteral",
    "UncertaintyAwareVoting",
    # Hyperdimensional
    "HyperdimensionalClauseMachine",
    "HDEncoder",
    "LevelHVEncoder",
    # Information Bottleneck
    "InformationBottleneckBinarizer",
    "InformationPreservingClauseMachine",
    "VIBLayer",
    # Hierarchical
    "HierarchicalMultiResolutionTM",
    "HierarchicalLevel",
    "CrossLevelAttention",
    # Attention Adaptive
    "NeuralSymbolicTransformer",
    "NeuralSymbolicBlock",
    "DynamicThresholdPredictor",
    # Enhanced Continuous Encoding
    "EnhancedContinuousTM",
    "EnhancedContinuousEncoder",
    "MultiScaleThermometer",
    "GaussianBasisExpansion",
    "LearnedFeatureBins",
    "PositionalValueEncoding",
    "LearnableBinarizer",
]

