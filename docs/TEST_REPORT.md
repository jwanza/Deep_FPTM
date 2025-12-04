# FuzzyPatternTM Test Report

## Test Suite Summary

**Total Tests: 456 passed, 1 skipped**

| Test Suite | Tests | Status |
|------------|-------|--------|
| `test_booleanization_unit.py` | 41 | ✅ PASS |
| `test_booleanization_e2e.py` | 15 | ✅ PASS |
| `test_integration.py` | 22 | ✅ PASS |
| `test_hyperbolic.py` | 50 | ✅ PASS |
| `test_sparse_routing.py` | 31 | ✅ PASS |
| `test_continual_learning.py` | 40+ | ✅ PASS |
| `test_stcm_unit.py` | 8 | ✅ PASS |
| `test_stcm_e2e.py` | 7 | ✅ PASS |
| Other test files | 240+ | ✅ PASS |

---

## Comprehensive Test Matrix

| Model Family | New Feature Coverage | Primary Tests / Scripts |
|--------------|----------------------|-------------------------|
| STE TM / STCM / FPTM-Equiv | Learnable binarizers, clause curriculum, KD hooks | `tests/test_binarizers.py`, `tests/test_schedule.py`, `experiments/run_validation_suite.py --suite mnist-smoke` |
| Deep TM / Deep STCM | Clause grouping, curriculum, KD, SAM/Lion toggles | `tests/test_schedule.py`, `run_mnist_equiv.py --models deep_tm,deep_stcm` |
| Deep CTM / Deep CSTCM | Multi-scale conv cores, EMA self-distill | `tests/test_schedule.py`, `tests/run_cifar_ensemble.py --models deep_ctm,deep_cstcm` |
| Hybrid CNN+TM + Learnable binarizers | Backbone factory, dual-sigmoid | `tests/test_backbone_factory.py`, `tests/test_binarizers.py`, `experiments/run_validation_suite.py --suite cifar-smoke --models hybrid` |
| Multi-scale TM Ensemble | Spatial attention, residual decision | `tests/test_spatial_tm.py`, `tests/test_swin_and_multiscale.py` |
| Swin / ViT Transformer Hybrids | Clause attention, KD/contrastive, auto clause tuning | `run_mnist_equiv.py --models transformer`, `tests/test_swin_and_multiscale.py`, `experiments/run_validation_suite.py --suite transformer-smoke` |

The matrix is mirrored in `docs/TEST_REPORT.md` so every new feature has at least one unit/component test plus an integration/smoke entry point.

---

## Dataset Targets & Metrics

| Dataset | Purpose | Baseline Target | Enhanced Target | Notes |
|---------|---------|-----------------|-----------------|-------|
| MNIST / Fashion-MNIST | Fast functional smoke for all variants | Match historical accuracy ±1% | Verify no regressions after refactors | Runs with batch 64, 1–2 epochs |
| CIFAR-10 | Primary accuracy benchmark | Baseline configs captured in `experiments/configs/*_baseline.json` | +2% Top-1 over baseline or ≥10% throughput gain | Logged via `results/cifar10/<model>.json` |
| CIFAR-10 (Nightly) | Extended runs for transformers + Swin hybrids | 60–65% (short run) | 70%+ with KD and binarizers | Executed via `experiments/run_validation_suite.py --suite cifar-full` |

Metrics tracked per run: top-1 accuracy, train throughput (samples/s), best-epoch accuracy, clause entropy, attention focus entropy, and CUDA max memory when available.

---

## CIFAR-10 Benchmark Results

### Baseline vs. Optimized

| Model | Test Accuracy | Epochs | Time | Notes |
|-------|---------------|--------|------|-------|
| **Deep-CSTCM (Baseline)** | **64.37%** | 5 | 225s | Original configuration |
| **Deep-CSTCM (Optimized)** | **77.00%** | 25 | 829s | +12.63% improvement |
| Deep-CSTCM (Self-distill) | 74.01% | 30 | 857s | Higher self-distillation weight |

### Best Configuration (77.00% Accuracy)

```bash
python3 python/fptm_ste/tests/run_mnist_equiv.py \
    --dataset cifar10 \
    --models deep_cstcm \
    --epochs 25 \
    --batch-size 128 \
    --lr 0.003 \
    --normalize auto \
    --deepctm-stem \
    --deepctm-stem-channels 64 \
    --deepctm-mix linear \
    --deepctm-channels 64,128,256 \
    --deepctm-kernels 3,3,3 \
    --deepctm-strides 1,1,1 \
    --deepctm-pools 2,2,2 \
    --deepctm-clauses 256,256,512 \
    --deepctm-head-clauses 1024 \
    --deepcstcm-core stcm \
    --deepctm-aux-weight 0.1 \
    --ctm-ema-decay 0.995 \
    --ctm-self-distill-weight 0.1 \
    --ctm-self-distill-temp 2.0
```

---

## New Modules Implemented

### Booleanization Solutions (python/fptm_ste/booleanization/)

1. **ContinuousResidualClauseMachine** - Dual-stream binary + continuous architecture
2. **ProbabilisticLiteralClauseMachine** - Distributional literals with uncertainty
3. **HyperdimensionalClauseMachine** - HD computing for similarity-preserving encoding
4. **InformationPreservingClauseMachine** - Information Bottleneck binarization
5. **HierarchicalMultiResolutionTM** - Multi-resolution clause hierarchy
6. **NeuralSymbolicTransformer** - Per-sample dynamic binarization
7. **EnhancedContinuousTM** - Multi-scale thermometer + Gaussian + Positional encoding

### Advanced Features

- **Hyperbolic Geometry** (`hyperbolic.py`): Poincare ball projection, hyperbolic voting
- **Sparse Routing** (`sparse_routing.py`): TopK router, L0 pruning, MoE
- **Continual Learning** (`continual.py`): EWC, SI, MAS, GEM, PackNet, Replay
- **LoRA Adapters** (`lora_adapter.py`): Low-rank adaptation for TM
- **SAM Optimizer** (`sam_optimizer.py`): Sharpness-aware minimization
- **Data Augmentation** (`augmentation.py`): Mixup, CutMix, ManifoldMixup
- **Temporal TM** (`temporal.py`): Sequence modeling with clauses
- **Ultimate Hybrid** (`ultimate_hybrid.py`): Combines all techniques

---

## Key Insights

### Why Deep-CSTCM Outperforms Flat Models

1. **Convolutional Architecture**: Preserves spatial structure of images
2. **Hierarchical Processing**: Multi-level feature extraction
3. **Self-Distillation**: Teacher-student learning within the model
4. **EMA Updates**: Exponential moving average for stable training

### Booleanization Bottleneck

The new booleanization modules (CRCM, HD, IB, etc.) achieve ~45-55% on CIFAR-10 with flattened features vs. 77% with convolutional architecture. The gap is due to:

- Loss of spatial structure when flattening 32x32x3 → 3072
- Single-threshold binarization losing continuous information
- No multi-scale feature extraction

### Future Improvements

1. Integrate booleanization with ConvTM architecture
2. Use pretrained CNN features before booleanization
3. Apply hierarchical spatial booleanization to patches

---

## Automation & CI Hooks

- **`python/fptm_ste/experiments/run_validation_suite.py`** orchestrates unit, MNIST-smoke, CIFAR-smoke, and CIFAR-full suites. Each suite streams logs to `logs/validation/<suite>.log` and emits JSON summaries.
- Fast suites (unit + MNIST-smoke) are wired for CI using `RUN_SUITE=unit,mnist-smoke python .../run_validation_suite.py`; heavier CIFAR suites are tagged for nightly builds.
- Suite definitions record command arguments so we can bisect regressions quickly and replay with exact seeds.

---

## Result Aggregation & Reproduction

- **Configs**: Reusable experiment descriptors live under `python/fptm_ste/experiments/configs/`. Each JSON file encodes dataset, model, baseline/enhanced flag, and CLI overrides for `tests/run_mnist_equiv.py`.
- **Execution**: `run_validation_suite.py --suite cifar-full --config-dir python/fptm_ste/experiments/configs` fans out runs across all configs (baseline first, enhanced second) and writes results to `results/cifar10/<model>/<variant>.json`.
- **Comparison**: `python/fptm_ste/experiments/analyze_results.py --baseline results/cifar10/deep_cstcm/baseline.json --enhanced results/cifar10/deep_cstcm/enhanced.json` prints accuracy/throughput deltas and highlights whether targets are met.
- **Reporting**: Aggregated CSV/Markdown tables are appended to this document after each sweep; automation scripts update clause/attention diagnostics to keep interpretability regressions visible.

---

## Test Commands

```bash
# Run all unit tests
pytest python/tests/test_booleanization_unit.py -v

# Run all E2E tests
pytest python/tests/test_booleanization_e2e.py -v

# Run integration tests
pytest python/tests/test_integration.py -v

# Run all tests except SOTA validation
pytest python/tests/ -v --ignore=python/tests/test_sota_validation.py

# Run CIFAR-10 benchmark
python3 python/fptm_ste/tests/run_mnist_equiv.py --dataset cifar10 --models deep_cstcm --epochs 25 ...
```

---

## Date: December 2, 2024
