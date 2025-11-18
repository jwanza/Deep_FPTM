# Setun–Ternary Clause Machine (STCM)

STCM generalizes the fuzzy-pattern Tsetlin Machine family by unifying the
per-feature literal controllers into ternary states that map directly to the
balanced Setun logic `{−1, 0, +1}`:

* `+1` → include the positive literal (`xᵢ = 1`)
* `−1` → include the inverse literal (`xᵢ = 0`)
* `0`  → ignore the feature

Each clause bank (positive and negative polarity) now stores a *single* ternary
mask per feature rather than two independent automata for positive and negative
versions. This halves the literal parameter count, enforces mutual exclusivity,
and makes the clause representation directly compatible with TCAM-style hardware
or other Setun-inspired accelerators.

## Architecture overview

### Input handling
`FuzzyPatternTM_STCM` reuses the existing `prepare_tm_input` pipeline, including
image reshaping, grayscale auto-expansion, and channel reductions. This keeps
the wiring consistent with every other TM variant in this repository and makes
the class drop-in compatible with current training scripts.

### Ternary masks and STE
Each clause bank owns a logit matrix `[clauses_half, n_features]`. The logits
are sent through a straight-through ternary quantizer:

```
t = tanh(logit / temperature)
hard =  1  if logit >  band
        -1 if logit < -band
         0 otherwise
mask = hard + (t - t.detach())   # STE
```

The neutral band avoids flapping between positive and negative literals and
provides a “don’t care” margin. Gradients flow through the soft `tanh` surrogate
while the forward pass uses the hardened ternary mask.

### Clause strength operators

Two operators are available:

1. **Capacity − mismatch (default)**  
   Matches the Julia-style FPTM logic. For each clause, the number of included
   literals becomes the capacity. The mismatch penalty counts literal violations
   using `F.linear` with the positive mask (on `1 − x`) and the inverse mask (on
   `x`). The clause output is `ReLU(capacity − mismatch)` with a straight-through
   gradient to keep the surface piecewise linear. `lf` and `literal_budget`
   constraints are applied before the capacity is broadcast.

2. **Product t-norm**  
   Computes penalty scores similar to `FuzzyPatternTM_STE`: the more literals
   that disagree with the input, the larger the exponential penalty. The final
   clause output is `exp(-clip(scale * penalties, 0, 10))`, preserving numerical
   stability even for long clauses.

Both operators return non-negative strengths for the positive bank and the
negatives are mirrored as `-strength` before voting, keeping the clause vector
structured as `[pos_strengths, -neg_strengths]`.

### Voting head

* **Continuous (default)**: learnable floating-point weights identical to the
  other TM modules.
* **Ternary voting (optional)**: an additional logit tensor shares the same STE
  ternarization logic, enabling ±1/0 votes for strict interpretability or
  hardware export. Switching between the two is a constructor flag; training can
  remain unchanged.

### Deep stacks
`DeepTMNetwork` now accepts a `layer_cls` parameter (defaulting to
`FuzzyPatternTM_STE`). When `layer_cls=FuzzyPatternTM_STCM`, the network will
instantiate STCM blocks for every hidden layer and for the classifier. Optional
`layer_operator` and `layer_ternary_voting` hints are added only if the chosen
class accepts those keyword arguments, preserving backward compatibility with
existing TM layers.

## Usage examples

```python
from fptm_ste import FuzzyPatternTM_STCM, DeepTMNetwork

# Standalone STCM with capacity operator and continuous voting
tm = FuzzyPatternTM_STCM(
    n_features=784,
    n_clauses=64,
    n_classes=10,
    operator="capacity",
    ternary_voting=False,
)

# Ternary voting + product operator
tm_tern = FuzzyPatternTM_STCM(
    n_features=128,
    n_clauses=32,
    n_classes=4,
    operator="product",
    ternary_voting=True,
    ternary_band=0.15,
)

# Deep stack using STCM blocks
deep_tm = DeepTMNetwork(
    input_dim=256,
    hidden_dims=[64, 64],
    n_classes=3,
    n_clauses=32,
    layer_cls=FuzzyPatternTM_STCM,
    layer_operator="capacity",
    layer_ternary_voting=False,
)
```

## Testing summary

| Test | Purpose |
| ---- | ------- |
| `python/tests/test_stcm_unit.py::test_stcm_forward_shapes_and_masks` | Shape sanity + mask range |
| `python/tests/test_stcm_unit.py::test_stcm_discretize_matches_logit_signs` | Ensures discretization mirrors ternary masks |
| `python/tests/test_stcm_unit.py::test_stcm_gradients_flow_through_masks` | Confirms STE gradients reach the logit tensors |
| `python/tests/test_stcm_unit.py::test_stcm_literal_parameters_are_halved` | Verifies literal parameter count reduction relative to FPTM |
| `python/tests/test_stcm_unit.py::test_capacity_and_product_operators_are_stable` | Checks both operators produce finite, well-formed clause outputs |
| `python/tests/test_stcm_e2e.py::test_stcm_learns_conjunctive_rule` | Demonstrates rapid convergence on a symbolic AND rule |
| `python/tests/test_stcm_e2e.py::test_capacity_and_product_improve_loss_on_fuzzy_data` | Shows both operators lower loss on fuzzy inputs |
| `python/tests/test_stcm_e2e.py::test_continuous_vs_ternary_voting_accuracy_gap_small` | Confirms ternary voting tracks continuous performance |
| `python/tests/test_stcm_e2e.py::test_deep_tm_network_uses_stcm_layers` | Validates deep stacks with STCM blocks train without instability |

## Practical tips

* **Band selection**: `ternary_band` controls the neutral zone. Start with
  `0.2–0.3` for balanced datasets; shrink it when you want more aggressive
  literal adoption.
* **Temperature**: `ste_temperature` tunes the slope of the surrogate tanh. Use
  `1.0` for most workloads; higher values smooth the gradients if training
  becomes jittery.
* **Budgets (`lf`, `literal_budget`)**: these still apply in capacity mode and
  keep clauses compact. In product mode they act as soft regularizers because
  the penalties accumulate faster for long clauses.
* **Export**: `discretize()` now returns `positive`, `positive_inv`, `negative`,
  `negative_inv`, along with metadata (`clauses_num`, `operator`,
  `ternary_voting`). Each list contains 1-based literal indices ready for JSON
  export.
* **Deep integration**: when combining with CNNs or transformers, place STCM
  blocks after the learnable backbone, just as you would with an STE TM. The
  ternary masks propagate gradients cleanly thanks to the STE surrogate.

With these pieces in place you can prototype Setun-style, clause-based models in
software while keeping a straight path toward hardware-efficient deployments.


