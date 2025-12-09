from .tm import (
    FuzzyPatternTM_STE,
    FuzzyPatternTMFPTM,
    FuzzyPatternTM_STCM,
    prepare_tm_input,
    # Advanced voting mechanisms
    AttentionVoting,
    HierarchicalVoting,
    ProbabilisticVoting,
    ConfidenceWeightedVoting,
    # Memory
    ClauseMemoryBank,
    ClauseMemoryAttention,
)
from .tm_feedback import EnhancedSTCM
from .tm_optimized import OptimizedSTCM
from .tm_integrated import IntegratedSTCM, build_stcm
from .backbones import (
    UniversalBackboneFactory,
    BackboneMetadata,
    get_backbone_normalization,
)
from .operators import (
    available_ternary_operators,
    build_ternary_operator,
    # Fuzzy t-norms
    LukasiewiczTNorm,
    GodelTNorm,
    HamacherProduct,
    YagerTNorm,
    DrasticProduct,
    EinsteinProduct,
    NilpotentMinimum,
    BoundedDifference,
    ParameterizedTNorm,
    SoftMinMax,
    AdaptiveOperatorMixer,
    EnsembleOperator,
)
from .deep_tm import DeepTMNetwork
from .swin_tm import (
    SwinTM,
    SwinTMStageConfig,
    build_swin_stage_configs,
    SwinFeatureExtractor,
    MultiScaleTMEnsemble,
)
from .attention_oracle import TM_Attention_Oracle
from .tm_transformer import TM_TransformerBlock, UnifiedTMTransformer
from .export import export_compiled_to_json
from .trainers import (
    anneal_ste_factor,
    train_step,
    ClauseCurriculumScheduler,
    ClauseMetricScheduler,
    ClauseContrastiveLoss,
    SupervisedContrastiveLoss,
    ClauseRepresentationLoss,
    train_epoch_with_curriculum,
)
from .resnet_tm import ResNetTM, resnet_tm18, resnet_tm34, resnet_tm50, resnet_tm101

# New modules
from .clause_attention import (
    HierarchicalClauseAttention,
    ClauseReasoningNetwork,
    ClauseTransformerBlock,
)
from .multires_tm import (
    MultiResolutionSTCM,
    AdaptiveThresholdSTCM,
    CascadeResolutionSTCM,
    HierarchicalResolutionSTCM,
    SpatialTMScaleConfig,
    SpatialTMEnsemble,
)
from .moe_tm import (
    SparseMoETM,
    BatchedSparseMoETM,
    HierarchicalMoETM,
    SwitchMoETM,
)
from .fusion_layers import (
    LearnableScaleAttention,
    ResidualInterpretableDecision,
)
from .pretraining import (
    MaskedClauseModeling,
    ContrastivePretraining,
    BYOLPretraining,
    ReconstructionPretraining,
    PretrainingWrapper,
    pretrain_tm,
)

# Augmentation utilities
from .augmentation import (
    mixup_data,
    cutmix_data,
    mixup_criterion,
    AugmentationPipeline,
)

# Incremental learning (Julia-style)
from .incremental_tm import (
    IncrementalConfig,
    TsetlinAutomaton,
    IncrementalSTCM,
    IncrementalDeepTM,
    incremental_train_step,
    incremental_train_epoch,
)

# Stable training utilities
from .stable_training import (
    StableTrainingConfig,
    StableEMA,
    AdaptiveLRScheduler,
    ConfidenceWeightedLoss,
    ClauseRegularizer,
    stable_train_step,
    stable_train_epoch,
    stable_evaluate,
    StableTrainer,
)

# Continual learning
from .continual import (
    ContinualLearningWrapper,
    EWCClauseMachine,
    EWCWrapper,
    SynapticIntelligenceClause,
    MemoryAwareSynapsesClause,
    ExperienceReplayBuffer,
    ReplayAugmentedTrainer,
    GradientEpisodicMemory,
    PackNetClause,
    ProgressiveClauseNetwork,
    ContinualLearningPipeline,
)

