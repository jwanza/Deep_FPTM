
import torch
from fptm_ste.swin_tm import SwinTM
from fptm_ste.resnet_tm import resnet_tm18
from fptm_ste.tm import FuzzyPatternTM_STCM

def demo_sota_stcm_swin():
    """
    Demonstrates the configuration of a SOTA-ready Setun-Ternary Clause Machine (STCM)
    integrated with a Swin Transformer backbone.
    
    SOTA Characteristics:
    1. Backbone: Swin Transformer (Hierarchical, Shifted Windows) captures spatial features efficiently.
    2. Logic Layer: STCM (Sparse Tsetlin Clause Machine) replaces standard dense layers.
       - Uses ternary masks {-1, 0, +1} for feature selection (0 = ignore).
       - Reduces parameters and noise compared to dense binary masks.
    3. Voting: Ternary Voting enabled.
       - Votes are {-1, 0, +1}, making the final decision fully interpretable/summable.
    4. Operator: 'capacity' (Setun logic).
    """
    print("=== Building SOTA STCM Swin Model ===")
    
    # Configuration for SOTA
    tm_kwargs = {
        "ternary_voting": True,    # Enable ternary voting for interpretability
        "operator": "capacity",    # Use capacity-mismatch operator (robust logic)
        "ternary_band": 0.1,       # Band for "ignore" state (0) in voting/masks
    }

    # Instantiate SwinTM with STCM
    # We use 'tiny' preset for efficiency, but 'small'/'base' can be used for higher accuracy.
    model = SwinTM(
        preset="tiny",
        num_classes=10,            # e.g., CIFAR-10
        in_channels=3,
        image_size=(224, 224),
        tm_cls="FuzzyPatternTM_STCM",
        tm_kwargs=tm_kwargs
    )

    print("\n[Model Structure]")
    print(model)

    # Verify that STCM layers are correctly instantiated
    print("\n[Verification]")
    stcm_layers = [m for m in model.modules() if isinstance(m, FuzzyPatternTM_STCM)]
    print(f"Number of STCM layers: {len(stcm_layers)}")
    
    if len(stcm_layers) > 0:
        first_layer = stcm_layers[0]
        print(f"Layer 0 Type: {type(first_layer).__name__}")
        print(f"Layer 0 Operator: {first_layer.operator}")
        print(f"Layer 0 Ternary Voting: {first_layer.ternary_voting}")
        
        # Check parameter efficiency
        # In STCM, we expect 2 ternary masks (pos, neg) of shape [clauses//2, features]
        # Instead of 4 binary masks in standard FPTM.
        print(f"Layer 0 Pos Logits Shape: {first_layer.pos_logits.shape}")
        # Standard FPTM would have ta_pos, ta_neg, ta_pos_inv, ta_neg_inv
        
        assert first_layer.ternary_voting is True, "Ternary voting should be enabled"
        print("Verification Successful: STCM properly integrated.")
    
    # Forward pass check
    x = torch.randn(2, 3, 224, 224)
    print("\n[Forward Pass]")
    logits, stage_logits, clause_lists, final_clauses = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print("Forward pass successful.")

def demo_sota_stcm_resnet():
    """
    Demonstrates ResNet + STCM.
    """
    print("\n=== Building SOTA STCM ResNet-18 Model ===")
    
    tm_kwargs = {
        "ternary_voting": True,
        "operator": "capacity",
    }
    
    model = resnet_tm18(
        num_classes=10,
        tm_cls=FuzzyPatternTM_STCM, # Can pass class directly
        tm_kwargs=tm_kwargs
    )
    
    stcm_layers = [m for m in model.modules() if isinstance(m, FuzzyPatternTM_STCM)]
    print(f"Number of STCM layers in ResNet: {len(stcm_layers)}")
    
    x = torch.randn(2, 3, 224, 224)
    logits, _, _, _ = model(x)
    print(f"ResNet Output shape: {logits.shape}")


if __name__ == "__main__":
    demo_sota_stcm_swin()
    demo_sota_stcm_resnet()

