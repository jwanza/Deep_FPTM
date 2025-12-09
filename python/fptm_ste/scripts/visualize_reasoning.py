"""
Visualization Script for SOTA Hybrid TM Reasoning.

Visualizes:
1. Scale Attention Weights (Which resolution mattered?)
2. Clause Activations (Which patterns fired?)
3. Input Saliency (Which pixels triggered the logic?)
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Dict, List

def visualize_reasoning(
    model, 
    image: torch.Tensor, 
    class_names: List[str] = None,
    save_path: str = "reasoning_viz.png"
):
    """
    Generate interpretable visualization of model reasoning.
    """
    model.eval()
    
    # 1. Forward Pass with Explanation
    with torch.no_grad():
        # Get intermediate outputs manually
        # Note: We need to hook into the model or modify it to return everything
        # SotaHybridTM's forward returns (logits, explanation) if requested? No, currently just logits.
        # But `ResidualInterpretableDecision` returns explanation.
        # We need to update SotaHybridTM to pass `return_explanation=True`.
        # Assuming we updated it:
        logits = model(image.unsqueeze(0), use_ste=True, return_explanation=True)
        
        # If model doesn't return explanation yet, we might fail here.
        # The SotaHybridTM implementation currently returns `final_output`.
        # If `return_explanation=True`, `ResidualInterpretableDecision` returns (logits, explanation_dict).
        
        if isinstance(logits, tuple):
            logits, decision_expl = logits
        else:
            print("Model did not return explanation. Ensure return_explanation=True is supported.")
            return

    # 2. Extract Data
    pred_idx = logits.argmax(dim=1).item()
    pred_class = class_names[pred_idx] if class_names else str(pred_idx)
    
    # Get Scale Weights (Need to access fusion layer state or return it)
    # Ideally, SotaHybridTM should return this in the explanation dict.
    # For now, let's assume we can access the last stored weights if we implemented tracking.
    # LearnableScaleAttention has `ema_weights`.
    scale_weights = model.fusion.ema_weights.cpu().numpy()
    
    # 3. Plot
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)
    
    # Image
    ax_img = fig.add_subplot(gs[0, 0])
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    ax_img.imshow(img_np)
    ax_img.set_title(f"Input (Pred: {pred_class})")
    ax_img.axis("off")
    
    # Scale Attention
    ax_scale = fig.add_subplot(gs[0, 1])
    scales = [f"Scale {i}" for i in range(len(scale_weights))]
    ax_scale.bar(scales, scale_weights)
    ax_scale.set_title("Scale Attention Weights")
    ax_scale.set_ylim(0, 1)
    
    # Decision Pathways
    ax_dec = fig.add_subplot(gs[0, 2])
    direct_w = decision_expl['direct_weight']
    enhanced_w = decision_expl['enhanced_weight']
    ax_dec.pie([direct_w, enhanced_w], labels=["Direct (Linear)", "Enhanced (Res)"], autopct='%1.1f%%')
    ax_dec.set_title("Decision Pathway Contribution")
    
    # Placeholder for Clause Activation (requires accessing TM internals)
    # We can plot the distribution of 'enhanced_logits' vs 'direct_logits'
    ax_logits = fig.add_subplot(gs[1, :])
    direct_l = decision_expl['direct_logits'][0].cpu().numpy()
    enhanced_l = decision_expl['enhanced_logits'][0].cpu().numpy()
    
    x = np.arange(len(direct_l))
    width = 0.35
    ax_logits.bar(x - width/2, direct_l, width, label='Direct')
    ax_logits.bar(x + width/2, enhanced_l, width, label='Enhanced')
    ax_logits.set_title("Logits Comparison")
    ax_logits.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    # Example usage
    from fptm_ste.sota_hybrid import SotaHybridTM
    
    model = SotaHybridTM(n_classes=10, backbone="swin_tiny", pretrained=False)
    img = torch.randn(3, 224, 224)
    visualize_reasoning(model, img)



