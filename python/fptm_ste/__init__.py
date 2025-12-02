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
from .tm_integrated import IntegratedSTCM, build_stcm
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
)
from .moe_tm import (
    SparseMoETM,
    BatchedSparseMoETM,
    HierarchicalMoETM,
    SwitchMoETM,
)
from .fusion_layers import (
    TMAttentionFusion,
    AdaptiveFusionBlock,
    DeepTMAttentionNetwork,
    HybridVisionTM,
)
from .pretraining import (
    MaskedClauseModeling,
    ContrastivePretraining,
    BYOLPretraining,
    ReconstructionPretraining,
    PretrainingWrapper,
    pretrain_tm,
)


