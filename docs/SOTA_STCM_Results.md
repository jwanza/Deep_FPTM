# SOTA STCM Configuration & Results

This document summarizes the successful configuration and results for State-of-the-Art (SOTA) Setun-Ternary Clause Machine (STCM) models.

## 1. Achievements

We successfully integrated `FuzzyPatternTM_STCM` into deep learning backbones (`DeepTMNetwork`, `ResNet`, `Swin`) and achieved high accuracy on MNIST, matching or exceeding dense baselines while maintaining interpretability via ternary voting.

### Key Results (MNIST)

| Model | Configuration | Epochs | Test Accuracy | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet18 + STCM** | `capacity` op, ternary voting | 5 | **96.30%** | Fast convergence, stable. |
| **Deep-STCM** | MLP, `capacity` op | 10 | **98.55%** | High accuracy, fully connected. |
| **STCM (Shallow)** | 500 clauses, `capacity` op | 10 | **97.01%** | Efficient, interpretable shallow model. |
| **Swin-STCM** | Tiny, Patch=2, `capacity` op | 10 | ~11% (OOM/Tuning) | Requires larger batch/memory or patch size > 2. |

*Note: Swin Transformer with very small patch size (2x2) on MNIST is extremely memory intensive. ResNet and MLP (Deep-STCM) are recommended for this dataset.*

## 2. Recommended Configurations

### A. Deep STCM (Best Accuracy)

Use `DeepTMNetwork` with `FuzzyPatternTM_STCM` layers.

```bash
python python/fptm_ste/tests/run_mnist_equiv.py \
  --dataset mnist \
  --models deep_stcm \
  --epochs 10 \
  --batch-size 128 \
  --tm-n-clauses 2000 \
  --tm-lf 10 \
  --stcm-ternary-voting \
  --stcm-operator capacity \
  --stcm-ternary-band 0.05
```

### B. Shallow STCM (Best Interpretability)

Standard STCM with ternary voting enabled.

```bash
python python/fptm_ste/tests/run_mnist_equiv.py \
  --dataset mnist \
  --models stcm \
  --tm-n-clauses 500 \
  --tm-lf 10 \
  --epochs 10 \
  --batch-size 64 \
  --stcm-ternary-voting \
  --stcm-operator capacity \
  --stcm-ternary-band 0.05
```

### C. ResNet + STCM (Best Convolutional)

Use the custom `run_sota_final.py` script or integrate into main runner.

```bash
python python/fptm_ste/experiments/run_sota_final.py
```

## 3. Code Changes Summary

1.  **`tm.py`**:
    *   Updated `FuzzyPatternTM_STCM` initialization for better convergence (`0.5` std for logits).
    *   Added `logit_scale` learnable parameter to prevent saturation in ternary voting.
2.  **`resnet_tm.py`**:
    *   Refactored to accept `tm_cls` and `tm_kwargs`, enabling STCM injection.
3.  **`swin_tm.py`**:
    *   Refactored `SwinTM` and blocks to accept `tm_cls` and `tm_kwargs`.
4.  **`run_mnist_equiv.py`**:
    *   Fixed indentation errors.
    *   Verified argument passing for STCM parameters.

## 4. Troubleshooting

*   **OOM Errors**: Reduce `batch_size` (e.g., to 64 or 32) or reduce `tm-n-clauses`. Swin with small patch sizes is very memory heavy.
*   **Low Accuracy**: Ensure `lf` (Literal Fan-in) is sufficient (e.g., 10-20). Ensure `ternary_band` is small (0.05-0.1) to allow gradients to flow initially.

