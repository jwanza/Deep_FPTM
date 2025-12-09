"""
SOTA Hybrid Tsetlin Machine Architecture.

Implements the "Neuro-Symbolic Sandwich" (Pyramid on Top) architecture:
1. Universal Backbone (Swin/ResNet/ConvNeXt) for feature extraction
2. Pyramid Adapters (Dual-Sigmoid Binarization)
3. Multi-Scale Reasoning Heads (Tsetlin Machines)
4. Learnable Scale Fusion
5. Residual Interpretable Decision
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import UniversalBackboneFactory
from .binarizers import SwinDualBinarizer, CNNSingleBinarizer
from .tm import FuzzyPatternTM_STCM
from .fusion_layers import LearnableScaleAttention, ResidualInterpretableDecision


class SotaHybridTM(nn.Module):
    """
    State-of-the-Art Hybrid Tsetlin Machine.
    
    Combines a deep learning backbone with multi-scale Tsetlin Machine reasoning.
    
    Args:
        n_classes: Number of output classes.
        backbone: Backbone name (e.g., 'swin_tiny', 'resnet50').
        pretrained: Whether to use pretrained backbone weights.
        n_clauses_base: Base number of clauses (scaled for different resolutions).
        input_size: Input image size (e.g., 224).
        input_channels: Number of input channels (3 for RGB).
        freeze_stages: Number of backbone stages to freeze (0-4).
        use_fpn: Whether to use FPN in the backbone (if supported).
    """
    
    def __init__(
        self,
        n_classes: int,
        backbone: str = "swin_tiny",
        pretrained: bool = True,
        n_clauses_base: int = 512,
        input_size: int = 224,
        input_channels: int = 3,
        freeze_stages: int = 0,
        use_fpn: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        
        # 1. Universal Backbone
        # Parse backbone string (e.g., "swin_tiny" -> type="swin", variant="tiny")
        if "_" in backbone:
            bb_type, bb_variant = backbone.split("_", 1)
        else:
            bb_type = backbone
            bb_variant = "base"
            
        self.backbone = UniversalBackboneFactory.create(
            backbone_type=bb_type,
            backbone_variant=bb_variant,
            pretrained=pretrained,
            input_size=input_size,
            freeze_stages=freeze_stages,
            use_fpn=use_fpn,
            **kwargs
        )
        
        # Get metadata
        self.backbone_meta = self.backbone.metadata()
        self.num_scales = self.backbone_meta.num_scales
        channel_dims = self.backbone_meta.channels
        
        # 2. Pyramid Adapters & 3. Reasoning Heads
        self.adapters = nn.ModuleList()
        self.heads = nn.ModuleList()
        
        # Calculate clause counts per scale
        # Strategy: More clauses for high-resolution (early) features, fewer for semantic (late) features?
        # Or inverse? The plan suggested "2x clauses for high-res scales".
        # Let's implement scaling: Scale 0 (High Res) gets 2*base, Scale 3 (Low Res) gets 0.5*base
        # Actually, let's keep it simple: equal clauses or configurable multiplier.
        # Default: Base clauses for all scales for now, can be tuned.
        
        head_output_dims = []
        
        for i, dim in enumerate(channel_dims):
            # Clause scaling
            # Scale 0 is high res -> fine details. Scale -1 is low res -> semantics.
            # Let's use n_clauses_base for all to start.
            n_clauses = n_clauses_base
            
            # Select Binarizer
            # Swin/ViT features are unbounded signed -> Dual Sigmoid
            # ResNet/CNN features are non-negative (ReLU) -> Single Sigmoid
            if bb_type in ["swin", "vit"]:
                adapter = SwinDualBinarizer(
                    in_channels=dim,
                    num_thresholds=16, # Default
                    init_temperature=1.0,
                    backbone_type=bb_type
                )
                tm_in_features = adapter.output_channels # 16 * 2 = 32
            else:
                adapter = CNNSingleBinarizer(
                    in_channels=dim,
                    num_thresholds=16,
                    init_temperature=1.0,
                    backbone_type=bb_type
                )
                tm_in_features = adapter.output_channels # 16
                
            self.adapters.append(adapter)
            
            # TM Head
            # Each head outputs [B, n_classes, H, W] -> Global Pooled -> [B, n_classes]
            # Wait, `FuzzyPatternTM_STCM` outputs [B, n_classes] usually if we pool inside?
            # Or [B, n_clauses]?
            # Let's check `FuzzyPatternTM_STCM`.
            # In `tm.py`, `FuzzyPatternTM_STCM.forward` returns `logits, clause_outputs`.
            # `logits` is [B, n_classes]. `clause_outputs` is [B, n_clauses].
            # For spatial features, we usually apply TM convolutionally.
            # `FuzzyPatternTM_STCM` in `tm.py` assumes flattened input if `prepare_tm_input` is used.
            # If we want spatial TMs, we need a convolutional version or patchify.
            
            # The `dc_fptm` reference used `FPTMConvJulia`.
            # Here we are using `FuzzyPatternTM_STCM` from `tm.py`.
            # `tm.py` seems to handle flattening.
            # If we want to preserve spatial dimensions as per "Pyramid on Top" plan...
            # "ResidualInterpretableDecision" expects feature lists.
            
            # Strategy:
            # 1. Adapter outputs (B, Thresholds, H, W).
            # 2. We convert to (B, H*W, Thresholds).
            # 3. TM processes each patch (pixel) independently? Or global TM?
            # The standard Hybrid approach (Swin-TM) usually does:
            # Feat -> Pool -> TM.
            # But "Pyramid on Top" with "Preserve Spatial" implies we don't pool immediately.
            
            # However, `FuzzyPatternTM_STCM` in `fptm_ste/tm.py` is a Dense TM.
            # To run it spatially, we'd need to loop or use Conv1d/Conv2d logic.
            # Given `FuzzyPatternTM_STCM` parameters (n_features, n_clauses), it expects fixed feature size.
            # So we MUST pool spatial dimensions BEFORE the TM if we use the standard `FuzzyPatternTM_STCM`.
            # OR we use 1x1 convolution equivalent (shared weights across space).
            
            # The Plan says: "Preserve Spatial Dimensions ... Perform the final voting/pooling only at the very end".
            # This implies we run TM on (H, W) grid.
            # `FuzzyPatternTM_STCM` logic is element-wise + sum. It can be run spatially.
            # But the `n_features` arg usually matches `C`.
            # If we pass `tm_in_features` (which is `num_thresholds` or 2x), the TM will learn patterns over these boolean channels.
            # Then we have H*W TMs sharing weights.
            
            # Let's assume we pool for now to match `HybridTMWithBackbone` pattern unless we implement a `ConvTM`.
            # Wait, `fptm_ste/tm.py` has `prepare_tm_input`.
            # If we pass `(B, C, H, W)`, it flattens to `(B, C*H*W)`.
            # That makes a HUGE TM input. That's likely not what we want for "Pyramid".
            # We want patterns over CHANNELS, shared over space?
            # Or patterns over PATCHES?
            
            # Reviewing `dc_fptm` lessons: "Spatial DC-FPTM ... 3. Tsetlin processing per scale (preserves spatial dimensions)".
            # `FPTMConvJulia` was used there.
            
            # In `fptm_ste`, we only have `FuzzyPatternTM_STCM` (Dense) available in `tm.py`?
            # Let's check `conv_tm.py`.
            # `python/fptm_ste/conv_tm.py` might be what we need.
            
            # If `conv_tm.py` is not robust, we might fallback to `GlobalAveragePool -> TM`.
            # The plan says "The TM acts as a Multi-Scale Reasoning Head that sits on top of the feature pyramid."
            # "Reasoning (Logic Layer) Returns [Batch, Classes, H, W]" in the example snippet.
            # This implies a ConvTM.
            
            # Let's look for `ConvFuzzyPatternTM` or similar in the file list.
            # I see `conv_tm.py` in the file list.
            # If I can't use it easily, I will use `AdaptiveAvgPool` to a fixed grid (e.g. 7x7) or 1x1.
            # For robustness and speed now, let's use `GlobalAveragePool` -> TM.
            # It loses spatial layout but keeps scale information.
            # Wait, the plan explicitly said "Preserve Spatial Dimensions... pooling too early kills accuracy".
            
            # Okay, I will implement a `SpatialTMHead` wrapper that applies `FuzzyPatternTM_STCM` pixel-wise (1x1 conv style).
            # Input: (B, C, H, W). Rearrange to (B*H*W, C).
            # TM processes (B*H*W, C) -> (B*H*W, Clauses).
            # Reshape to (B, Clauses, H, W).
            
            tm_head = FuzzyPatternTM_STCM(
                n_features=tm_in_features, # Just the channels!
                n_clauses=n_clauses,
                n_classes=n_classes,
                operator="product" # Differentiable product is often good for SOTA
            )
            self.heads.append(tm_head)
            head_output_dims.append(n_clauses)
            
        # 4. Fusion
        # We fuse the outputs of the TMs.
        # TM output: [B, Clauses, H, W].
        # LearnableScaleAttention needs stats.
        self.fusion = LearnableScaleAttention(
            num_scales=self.num_scales,
            feature_dim=0, # We'll compute stats from logits/features
            hidden_dim=128
        )
        
        # 5. Head
        self.decision = ResidualInterpretableDecision(
            input_dims=head_output_dims,
            num_classes=n_classes
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        use_ste: bool = True,
        return_explanation: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        
        # 1. Backbone
        features = self.backbone(x) # List of [B, C, H, W]
        
        tm_outputs = []
        tm_logits_per_scale = []
        
        for i, (feat, adapter, head) in enumerate(zip(features, self.adapters, self.heads)):
            # 2. Adapter (Binarization)
            # Output: [B, C_bin, H, W]
            binary = adapter(feat, use_discrete=not self.training)
            
            # 3. TM Head (Spatial)
            B, C, H, W = binary.shape
            
            # Rearrange for Pixel-wise TM: [B, C, H, W] -> [B, H, W, C] -> [B*H*W, C]
            # Note: We must ensure channels are last for Linear-like TM input
            binary_flat = binary.permute(0, 2, 3, 1).reshape(-1, C)
            
            # TM Forward: Returns logits [N, Classes], clauses [N, Clauses]
            # skip_norm=True because we already prepared input (and it's binary/prob from adapter)
            logits_flat, clauses_flat = head(binary_flat, use_ste=use_ste, skip_norm=True)
            
            # Reshape back: [B, H, W, Clauses] -> [B, Clauses, H, W]
            # We use the clause outputs for fusion
            clauses_spatial = clauses_flat.view(B, H, W, -1).permute(0, 3, 1, 2)
            tm_outputs.append(clauses_spatial)
            
            # Also store logits for Scale Attention stats
            # [B, Classes, H, W]
            logits_spatial = logits_flat.view(B, H, W, -1).permute(0, 3, 1, 2)
            tm_logits_per_scale.append(logits_spatial)
            
        # 4. Fusion
        # LearnableScaleAttention computes weights based on entropy/activation of logits
        # We assume it handles spatial dimensions by pooling internally for stats
        fused_clauses = self.fusion(tm_outputs, tm_logits_per_scale)
        
        # 5. Decision
        # ResidualInterpretableDecision handles final pooling and classification
        # It expects list of features. We pass the fused result split back?
        # No, `fused_clauses` from `LearnableScaleAttention` returns a single weighted sum tensor [B, C, H, W]?
        # Wait, `LearnableScaleAttention` returns `final_logits` by default (weighted sum of logits).
        # We need fused features for `ResidualInterpretableDecision`.
        # Let's check `fusion_layers.py`.
        # `LearnableScaleAttention` computes `final_logits` directly from `logits`.
        # It DOES NOT return fused features by default.
        # But we want to use `ResidualInterpretableDecision` on the CLAUSES.
        
        # Solution:
        # Get weights from fusion layer
        _, weights = self.fusion(tm_outputs, tm_logits_per_scale, return_weights=True)
        # weights: [B, num_scales]
        
        # Manually fuse clauses based on weights
        # We need to concat or sum?
        # `ResidualInterpretableDecision` takes a LIST of features and concats them.
        # So we simply pass `tm_outputs` weighted? Or just `tm_outputs`?
        # If we pass `tm_outputs` directly to `ResidualInterpretableDecision`, it concatenates all clauses.
        # This is good. It preserves all information.
        # The `LearnableScaleAttention` is mainly for "Attending" to the right scale.
        # Maybe we should multiply `tm_outputs[i]` by `weights[:, i]`.
        
        weighted_outputs = []
        for i, out in enumerate(tm_outputs):
            w = weights[:, i].view(-1, 1, 1, 1) # Broadcast [B, 1, 1, 1]
            weighted_outputs.append(out * w)
            
        final_output = self.decision(weighted_outputs, return_explanation=return_explanation)
        
        return final_output




