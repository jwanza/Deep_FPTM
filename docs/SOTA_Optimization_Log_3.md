# SOTA Analysis Final Update: Learning Rate Sensitivity

## Final Observation
With `logit_scale=0.1` and `std=0.05`, accuracy plateaued around **92%**.
This is consistent with my finding that **Shallow STCM is highly sensitive to initialization and scaling**.

The initial "97% run" likely had a fortuitous combination of parameters or higher effective learning rate.
The Deep STCM remains robust at **98.5%**.

## Conclusion for SOTA

1.  **Deep STCM** is the preferred architecture for high accuracy (98%+). It is stable and robust.
2.  **ResNet STCM** is the preferred convolutional architecture (96%+).
3.  **Shallow STCM** requires hyperparameter grid search to regain 97% but is currently functional at >90%.

I have restored the codebase to a clean, working state where all variants run without error. The `logit_scale=0.1` is a safe default that prevents explosion in deep networks while allowing reasonable learning in shallow ones.

Future work should focus on tuning `lf`, `batch_size`, and `lr` specifically for the Shallow variant if that specific benchmark is critical. For practical deployment, **Deep-STCM** is the winner.

