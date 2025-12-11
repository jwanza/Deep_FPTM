import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Sequence, Type

# Import existing STCM machinery
from fptm_ste.tm import FuzzyPatternTM_STCM, _ste_ternary
from fptm_ste.conv_tm import ConvSTCM2d, ConvTM2dOptimized

class STCMConv2d(nn.Module):
    """
    STCM-Conv2D: A drop-in replacement for standard Conv2d but using
    Tsetlin Machine logic (Set Tsetlin Clause Machine).
    
    This module implements the "Convolutional Tsetlin Machine" logic where
    each "filter" is a Clause that scans the image.
    
    In PyTsetlinMachineCUDA (MultiClassConvolutionalTsetlinMachine2D), 
    patches are extracted and fed to a Tsetlin Machine.
    Here, we do this differentiably:
    1. Input X is unfolded into patches (or conv2d is used directly).
    2. Weights are Ternary (-1, 0, 1) learned via STE.
    3. Activation is Logic-based (Capacity or Product).
    4. Output is feature map of "Clause Strengths".
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int, # number of clauses
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        ternary_band: float = 0.05,
        ste_temperature: float = 1.0,
        ste_gradient_mode: str = "gated_linear",
        operator: str = "capacity",
        clause_dropout: float = 0.0,
    ):
        super().__init__()
        # Internally use the optimized ConvSTCM2d which is already 
        # a differentiable STCM convolution.
        
        # Note: ConvSTCM2d expects 'n_clauses' which matches 'out_channels'.
        # However, standard Conv2d maps In->Out.
        # In TM terms, each "Output Channel" is a Clause.
        # So n_clauses = out_channels.
        # And n_classes is essentially the number of voting outputs, 
        # but here we are just a feature extractor (layer), so we don't vote yet.
        # Wait, ConvSTCM2d in fptm_ste is designed to output VOTES for classes?
        # Let's inspect ConvSTCM2d in conv_tm.py.
        
        # ConvSTCM2d __init__:
        #   in_channels, out_channels (classes), kernel_size...
        #   n_clauses (total clauses used internally)
        
        # It seems ConvSTCM2d combines "Clause Generation" AND "Voting" into one layer
        # to produce Class Logits map [B, Classes, H, W].
        
        # But for Deep STCM, we want layers that output FEATURES (Clauses), not Class Logits.
        # We want Layer 1 -> Clauses -> Layer 2 -> Clauses ...
        
        # So we need a version that does NOT vote, just outputs clause strengths.
        
        self.conv = ConvSTCM2d(
            in_channels=in_channels,
            out_channels=out_channels, # This usually means 'classes' in ConvSTCM2d, we need to bypass voting
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            n_clauses=out_channels, # Set clauses = out_channels
            ternary_band=ternary_band,
            ste_temperature=ste_temperature,
            ste_gradient_mode=ste_gradient_mode,
            operator=operator,
            clause_dropout=clause_dropout,
            # We need to disable voting projection
        )
        
        # We need to patch ConvSTCM2d or create a new class 'STCMFeatureExtractor'
        # that returns the raw clause outputs.
        # Looking at ConvTM2dOptimized.forward:
        # It computes 'clause_outputs' [B, Clauses, H, W]
        # Then does 'F.conv2d(clause_outputs, w_vote)' to get [B, Classes, H, W]
        
        # We want just 'clause_outputs'.
        
    def forward(self, x, use_ste=True):
        # We subclassed/wrapped ConvSTCM2d.
        # Call with return_clauses=True to get [B, C_out, H, W]
        return self.conv(x, use_ste=use_ste, return_clauses=True)

class DeepSTCMConv2d(nn.Module):
    """
    Deep Stack of STCM Convolutional Layers.
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        channels: Sequence[int], # [64, 128, 256] -> Number of Clauses per layer
        kernels: Sequence[int],
        strides: Sequence[int],
        ternary_band: float = 0.05,
        ste_gradient_mode: str = "gated_linear",
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        
        curr_c = in_channels
        for c, k, s in zip(channels, kernels, strides):
            # Each layer is a STCM Conv
            layer = STCMConv2d(
                in_channels=curr_c,
                out_channels=c,
                kernel_size=k,
                stride=s,
                ternary_band=ternary_band,
                ste_gradient_mode=ste_gradient_mode
            )
            self.layers.append(layer)
            curr_c = c # Output of layer i is input to i+1
            
            # Batch Norm or Tsetlin Norm?
            self.layers.append(nn.BatchNorm2d(c))
            
        # Final Classification Head
        # Global Pooling -> Linear Voting
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(curr_c, num_classes)
        
    def forward(self, x, use_ste=True):
        for layer in self.layers:
            if isinstance(layer, STCMConv2d):
                x = layer(x, use_ste=use_ste)
            else:
                x = layer(x)
        
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return x

# Note: This file relies on updating conv_tm.py to support 'return_clauses'.
# I will update conv_tm.py next.

