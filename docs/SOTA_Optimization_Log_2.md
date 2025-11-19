# SOTA Analysis Update: Logit Scale 1.0

## Observation

With `logit_scale=1.0` and `std=0.05`, accuracy was **89.93%** at epoch 5, with very high initial loss (`20.4`).
This indicates **gradient explosion** or instability due to large logits.
The shallow STCM output can be large (summing up to 2000 votes).
`scale=1.0` means we feed values like `500` into softmax.

### The "Sweet Spot"

We need a scale that is **not too small** (dampens signal) and **not too large** (saturates softmax).
Original successful run likely relied on implicit scaling or specific initialization dynamics that I altered.

### Hypothesized Optimal Scale
`1.0 / sqrt(n_clauses)` was `~0.02`. Too small.
`1.0` is too large.
`0.1` might be the sweet spot.
`0.1 * 2000` = `200` max logit. Still large but manageable.
`0.05 * 2000` = `100`.

Let's try initializing `logit_scale` to `0.1`.

## Plan
1. Change `logit_scale` init to `0.1`.
2. Re-run Shallow STCM verification.

