from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
except ImportError:  # pragma: no cover
    matplotlib = None
    plt = None
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .tm_transformer import UnifiedTMTransformer


def _denormalize_image(
    image: torch.Tensor,
    mean_std: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
) -> np.ndarray:
    img = image.detach().cpu().float()
    if mean_std is not None:
        mean, std = mean_std
        mean_t = torch.tensor(mean, dtype=img.dtype, device=img.device).view(-1, 1, 1)
        std_t = torch.tensor(std, dtype=img.dtype, device=img.device).view(-1, 1, 1)
        img = img * std_t + mean_t
    img = img.clamp(0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


def _resize_heatmap(heatmap: torch.Tensor, target_hw: Tuple[int, int]) -> np.ndarray:
    if heatmap.ndim == 2:
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    elif heatmap.ndim == 3:
        heatmap = heatmap.unsqueeze(0)
    heatmap = F.interpolate(heatmap, size=target_hw, mode="bilinear", align_corners=False)
    heatmap = heatmap.squeeze().detach().cpu().numpy()
    return heatmap


def _safe_norm(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.linalg.norm(tensor, dim=dim)


def _aggregate_attention_vit(
    model: UnifiedTMTransformer,
    sample_index: int,
    image_size: Tuple[int, int],
    patch_size: int,
) -> Optional[torch.Tensor]:
    if not hasattr(model, "blocks") or not model.blocks:
        return None
    block = model.blocks[-1]
    head_ctx = getattr(block, "last_head_context", None)
    if head_ctx is None:
        return None
    scores = _safe_norm(head_ctx[sample_index], dim=-1)  # [heads, tokens]
    scores = scores.mean(dim=0, keepdim=True)  # [1, tokens]
    grid_h = image_size[0] // patch_size
    grid_w = image_size[1] // patch_size
    if scores.shape[-1] == grid_h * grid_w + 1:  # CLS token present
        scores = scores[:, 1:]
    scores = scores.view(1, grid_h, grid_w)
    scores = scores / (scores.max() + 1e-6)
    return scores.squeeze(0)


def _aggregate_clause_vit(
    model: UnifiedTMTransformer,
    sample_index: int,
    image_size: Tuple[int, int],
    patch_size: int,
) -> Optional[torch.Tensor]:
    block = model.blocks[-1]
    clause_outputs = getattr(block.ffn, "last_clause_outputs", None)
    if clause_outputs is None:
        return None
    clause_scores = clause_outputs[sample_index].mean(dim=-1)
    grid_h = image_size[0] // patch_size
    grid_w = image_size[1] // patch_size
    if clause_scores.shape[0] == grid_h * grid_w + 1:  # strip CLS token if present
        clause_scores = clause_scores[1:]
    clause_scores = clause_scores.view(grid_h, grid_w)
    clause_scores = clause_scores.abs()
    clause_scores = clause_scores / (clause_scores.max() + 1e-6)
    return clause_scores


def _aggregate_attention_swin(
    model: UnifiedTMTransformer,
    sample_index: int,
) -> Optional[torch.Tensor]:
    if not hasattr(model, "stages") or not model.stages:
        return None
    stage = model.stages[-1]
    block = stage.blocks[-1]
    head_tokens = getattr(block, "last_head_tokens", None)
    hw = getattr(block, "last_hw", None)
    if head_tokens is None or hw is None:
        return None
    H, W = hw
    scores = _safe_norm(head_tokens[sample_index], dim=-1).mean(dim=0)  # [tokens]
    scores = scores.view(H, W)
    scores = scores / (scores.max() + 1e-6)
    return scores


def _aggregate_clause_swin(
    model: UnifiedTMTransformer,
    sample_index: int,
) -> Optional[torch.Tensor]:
    stage = model.stages[-1]
    block = stage.blocks[-1]
    clause_outputs = getattr(block.ffn, "last_clause_outputs", None)
    hw = getattr(block, "last_hw", None)
    if clause_outputs is None or hw is None:
        return None
    H, W = hw
    clause_scores = clause_outputs[sample_index].mean(dim=-1)
    clause_scores = clause_scores.view(H, W)
    clause_scores = clause_scores.abs()
    clause_scores = clause_scores / (clause_scores.max() + 1e-6)
    return clause_scores


def _render_overlay_figure(
    image_np: np.ndarray,
    attn_map: Optional[torch.Tensor],
    clause_map: Optional[torch.Tensor],
    class_probs: torch.Tensor,
    class_names: Sequence[str],
    topk: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for idx, ax in enumerate(axes):
        if idx != 3:
            ax.axis("off")

    axes[0].imshow(image_np)
    axes[0].set_title("Original")

    if attn_map is not None:
        attn_np = attn_map.detach().cpu()
        attn_np = _resize_heatmap(attn_np, (image_np.shape[0], image_np.shape[1]))
        axes[1].imshow(image_np)
        axes[1].imshow(attn_np, cmap="magma", alpha=0.45)
        axes[1].set_title("Attention Heatmap")
    else:
        axes[1].set_title("Attention N/A")

    if clause_map is not None:
        clause_np = clause_map.detach().cpu()
        clause_np = _resize_heatmap(clause_np, (image_np.shape[0], image_np.shape[1]))
        axes[2].imshow(image_np)
        axes[2].imshow(clause_np, cmap="viridis", alpha=0.45)
        axes[2].set_title("Clause Activation")
    else:
        axes[2].set_title("Clause Activation N/A")

    probs = torch.softmax(class_probs, dim=-1)
    k = min(topk, probs.numel())
    values, indices = torch.topk(probs, k)
    labels = [class_names[idx] if idx < len(class_names) else f"class {idx}" for idx in indices.tolist()]
    bars = axes[3].barh(range(k), values.cpu().numpy()[::-1], color="#4C72B0")
    axes[3].set_yticks(range(k))
    axes[3].set_yticklabels(labels[::-1])
    axes[3].invert_yaxis()
    axes[3].set_xlim(0.0, 1.0)
    axes[3].set_xlabel("Probability")
    axes[3].set_title("Top Predictions")
    for bar, value, label in zip(bars, values.cpu().numpy()[::-1], labels[::-1]):
        axes[3].text(
            value + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{label} ({value:.2f})",
            va="center",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def generate_transformer_overlay(
    model: UnifiedTMTransformer,
    image: torch.Tensor,
    label: int,
    class_names: Sequence[str],
    architecture: str,
    patch_size: int,
    image_size: Tuple[int, int],
    mean_std: Optional[Tuple[Sequence[float], Sequence[float]]],
    output_path: Path,
    *,
    device: torch.device,
    topk: int = 3,
) -> None:
    was_training = model.training
    model.eval()
    if plt is None:
        raise RuntimeError("matplotlib is required for overlay visualization but is not installed.")

    img_batch = image.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_batch, use_ste=True, collect_diagnostics=True)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
    if was_training:
        model.train()

    sample_index = 0
    architecture = architecture.lower()
    if architecture == "vit":
        attn_map = _aggregate_attention_vit(model, sample_index, image_size, patch_size)
        clause_map = _aggregate_clause_vit(model, sample_index, image_size, patch_size)
    else:
        attn_map = _aggregate_attention_swin(model, sample_index)
        clause_map = _aggregate_clause_swin(model, sample_index)

    image_np = _denormalize_image(image, mean_std)
    generate_clause = clause_map if clause_map is not None else None
    _render_overlay_figure(
        image_np,
        attn_map,
        generate_clause,
        logits.squeeze(0),
        class_names,
        topk,
        output_path,
    )


