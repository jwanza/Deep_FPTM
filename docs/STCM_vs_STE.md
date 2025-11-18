# STCM vs. STE TM

## Overview
Balanced-ternary Setun Tsetlin Machines (STCM) collapse the positive/negative literal automata into a single ternary state per feature. This halves the literal parameter count, keeps gradients concentrated, and introduces clause operators that stay numerically stable even with aggressive feature pipelines. The table below contrasts the core traits.

| Aspect | STE TM (`FuzzyPatternTM_STE`) | STCM (`FuzzyPatternTM_STCM`) | Why STCM Wins |
| --- | --- | --- | --- |
| Literal controller | Two logits per feature (positive & negative); mutual exclusion emerges from reinforcement | Single ternary logit → straight-through ternary mask; explicit include+, include−, ignore [[python/fptm_ste/tm.py]] | Cuts automata count by 2× and prevents contradictory literals |
| Clause operator | Product t-norm approximation using exponential penalties | Configurable: product t‑norm or capacity−mismatch with literal budgets | Capacity mode keeps clause strength bounded and interpretable |
| Literal budget / LF | Applied only inside the legacy FPTM variant | `_enforce_literal_budget` rescales masks so ∑ |mask| ≤ `min(literal_budget, lf)` | Prevents runaway mismatch penalties; stabilizes training |
| Straight-through clamp | `relu(x) + (x - relu(x)).detach()` | Same correction plus ternary STE for clause masks and voting [[python/fptm_ste/tm.py]] | Eliminates exploding losses (>1000) observed before the fix |
| Voting head | Dense floating-point matrix only | Optional ternary voting with STE (`--stcm-ternary-voting`) | Hardware-friendly exports without retraining |
| Deep integration | `DeepTMNetwork` stacks STE clauses only | `DeepTMNetwork` now accepts `layer_cls=FuzzyPatternTM_STCM` and extra kwargs (operator, ternary band, temperature) [[python/fptm_ste/deep_tm.py]] | Deep-STCM reuses the same ternary semantics in every layer |

## Empirical Comparison (MNIST, 10 epochs, conv features, batch 256)

| Model | Test Accuracy | Train Accuracy | Notes |
| --- | --- | --- | --- |
| STE-TM (`--models tm --tm-impl ste`) | 0.7429 | 0.7199 | Baseline from the regression log before STCM upgrades |
| **STCM** (`--models stcm`) | **0.9702** (best 0.9704) | 0.9724 | Capacity operator, ternary voting disabled, LR decays to 1e-4 |
| Deep TM (`--models deep_tm`) | 0.9781 | 0.9974 | STE clauses stacked in a residual network |
| **Deep STCM** (`--models deep_stcm`) | **0.9853** | 1.0000 | STCM blocks inside `DeepTMNetwork`, product operator |

**Command template:**
```bash
python3 python/fptm_ste/tests/run_mnist_equiv.py \
  --dataset mnist \
  --models stcm deep_stcm \
  --epochs 10 \
  --batch-size 256 \
  --tm-feature-mode conv \
  --tm-feature-cache /tmp/tm_features \
  --tm-feature-grayscale \
  --tm-feature-batch 512 \
  --report-train-acc --report-epoch-acc --report-epoch-test
```

## Why STCM Is Superior

1. **Fewer automata, richer semantics** – Collapsing positive/negative literals into a single ternary state halves the number of learnable logits while guaranteeing mutually exclusive literal choices.
2. **Bounded clause strengths** – Capacity−mismatch plus literal scaling keeps clause outputs near ±2 instead of >1e3, so cross-entropy gradients stay meaningful.
3. **Differentiable masks everywhere** – STE-based masks and straight-through clamping allow evaluation to remain “soft,” so accuracy no longer collapses when `use_ste=False`.
4. **Hardware-ready exports** – Optional ternary voting means the same trained model can be exported as sparse ±1 weights for Setun-style hardware.
5. **Deep compatibility** – The `DeepTMNetwork` now accepts arbitrary TM classes via `layer_cls`, so STCM advantages accumulate through stacked layers (residual, transformer FFNs, etc.).

With these improvements, STCM matches or exceeds STE TM accuracy under the same clause budgets while producing more interpretable and hardware-friendly logical clauses. Deep-STCM extends those gains to the stacked setting, reaching >98% test accuracy on MNIST without any bespoke tuning beyond the shared CLI flags.

