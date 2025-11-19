# SOTA Verification: Deep-STCM Stability

## Result
Deep-STCM ran successfully for **10 epochs** (single pass, no restarts) and achieved **93.93%** Test Accuracy.

This is lower than the **98.5%** previously seen.
Why?
Likely because I changed `FuzzyPatternTM_STCM` initialization (std=0.5) and added `logit_scale` to fix the Shallow model, but then reverted initialization to `0.05` but kept `logit_scale` at `0.1`.

### Analysis
*   **Deep-STCM** (MLP) benefits from higher variance init or standard init.
*   `logit_scale` of `0.1` might be too aggressive for Deep STCM if it relies on large logits for `LayerNorm`.

### Recommendation
To get back to 98%+, we should probably revert `logit_scale` for Deep STCM or tune `lf`.
Or simply use the `ResNet-STCM` which reached 96% efficiently.

But the critical task was fixing the loop restart and errors, which is **DONE**.
The infrastructure is now stable for hyperparameter tuning.

I will now generate the requested table and documentation.

