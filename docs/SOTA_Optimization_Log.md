# SOTA Analysis Update: Initialization Impact

## Observation

Reverting `torch.randn(...) * 0.5` to `* 0.05` (original) did **not** fully restore the 97% accuracy for shallow STCM in 5 epochs (reached 92.20%).

### Root Cause Analysis

The scaling factor (`self.logit_scale`) I introduced is `1.0 / sqrt(n_clauses)`.
For 2000 clauses, `scale` is `1/44.7 = 0.022`.
This aggressively dampens the logits fed into CrossEntropyLoss.
If logits are small, softmax probabilities are close to uniform, and gradients are small.

In the original high-performing run (97%), there was **no logit scaling**.
The votes were summed directly: `logits = biased @ voting`.
With `ternary_voting=True`, votes are `{-1, 0, 1}`.
Max possible logit = `2000` (all clauses vote +1).
CrossEntropyLoss on logits `[-2000, 2000]` is fine because softmax handles large values (hard max).
However, in deep networks, this large magnitude can cause exploding gradients or instability in subsequent layers (LayerNorm etc).

### Optimization Strategy

1.  **Shallow STCM**: Should NOT use `logit_scale` or should start with `scale=1.0`.
2.  **Deep STCM**: Benefits from `logit_scale` to keep activations normalized.

I will modify `FuzzyPatternTM_STCM` to make `logit_scale` optional or initialize it to 1.0 by default if ternary voting is on, but allow it to learn.

Actually, `0.05` initialization for logits + `logit_scale` = double dampening.
If I remove `logit_scale` (or set init to 1.0), the shallow model should recover.

Let's change `logit_scale` initialization to `0.1` (instead of `0.02`) or just `1.0`.
Or better: make it a keyword argument `logit_scale_init`.

Let's try changing `logit_scale` to initialize at `1.0`.

