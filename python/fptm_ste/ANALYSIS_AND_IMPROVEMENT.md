# Analysis and Improvement of STCM

## Deep Analysis of Set Tsetlin Convolutional Machine (STCM)

The STCM architecture represents a significant evolution of the Tsetlin Machine, bridging the gap between logic-based learning and differentiable deep learning.

### 1. Architectural Principles
STCM replaces the traditional boolean clause (which uses separate include/exclude bit-vectors) with a **Ternary Clause** structure.
- **Ternary State:** Each feature in a clause is in one of three states:
    1.  **Requirement 1 (Pos):** Feature must be High (1).
    2.  **Requirement 0 (Inv):** Feature must be Low (0).
    3.  **Ignore (Wildcard):** Feature value doesn't matter.
- **Differentiability:** Unlike standard Tsetlin Automata which update via discrete increments/decrements, STCM uses a **Straight-Through Estimator (STE)** or Softmax/Tanh relaxation to learn the ternary states via gradient descent.
- **Operators:**
    - **Capacity Operator:** Functions like a fuzzy counter. It counts satisfied literals against a capacity threshold. This creates a robust, noise-tolerant decision boundary.
    - **Product Operator:** Functions like a soft AND gate (conjunction).

### 2. Mathematical Equivalence and Optimization
Our deep analysis revealed that the STCM forward pass, despite its logical formulation, is mathematically equivalent to a **Constrained Sparse Linear Layer**.

**Original Formulation:**
Let $x$ be the input and $M_{pos}, M_{inv}$ be the ternary mask components ($M_{pos}=1$ if feature required 1, $M_{inv}=1$ if feature required 0).
The "Mismatch" (penalty) is calculated as:
$$ \text{Mismatch} = \sum (M_{pos} \cdot (1-x) + M_{inv} \cdot x) $$

**Optimized Formulation:**
We simplified this to:
$$ \text{Mismatch} = \sum M_{pos} - x \cdot (M_{pos} - M_{inv})^T $$
Let $W_{eff} = M_{pos} - M_{inv}$. This $W_{eff}$ takes values in $\{-1, 0, 1\}$.
$$ \text{Mismatch} = \text{Bias}_{pos} - x \cdot W_{eff}^T $$

This transformation:
1.  **Reduces Memory:** Eliminates the need to construct the `[1-x, x]` concatenated input (size $2F$).
2.  **Reduces Compute:** Reduces the matrix multiplication dimensions from $B \times 2F$ to $B \times F$, effectively **doubling the theoretical throughput** of the clause evaluation step.

We implemented this optimization in `tm_optimized.py` as `OptimizedSTCM`.

### 3. Comparison: STCM vs. ANN vs. SOTA Tsetlin

| Feature | Artificial Neural Network (ANN) | SOTA Tsetlin Machine (e.g., CUDA TM) | Optimized STCM (This Work) |
| :--- | :--- | :--- | :--- |
| **Weights** | Float32 (Continuous) | Bits (Binary/Integer states) | Ternary ({-1, 0, 1}) / Soft-Ternary |
| **Training** | Backpropagation (SGD/Adam) | Tsetlin Automata (RL-like) | Backpropagation (STE) |
| **Inference Compute** | Dense Float32 MatMul | Bitwise Operations (XOR, AND, POPCOUNT) | Sparse/Ternary MatMul |
| **Interpretability** | Low (Black Box) | High (Propositional Logic) | High (Logical Rules) |
| **Sparsity** | Typically Dense | Intrinsically Sparse | Intrinsically Sparse |
| **Hardware Fit** | GPU/TPU (Tensor Cores) | CPU/FPGA (Bitwise logic) | GPU (Tensor Cores) or CPU |
| **Accuracy** | SOTA on Perception | Strong on Tabular/Logic, catching up on Vision | Competitive Hybrid |

**Key Advantages of STCM:**
1.  **Differentiable Logic:** Allows STCM to be dropped into any PyTorch pipeline (CNNs, Transformers) as a layer, which is difficult with standard Tsetlin Machines.
2.  **GPU Utilization:** Unlike bitwise TMs which require custom CUDA kernels to be fast, STCM utilizes standard highly-optimized Matrix Multiplication (GEMM) routines, while still converging to sparse, logical representations.
3.  **Efficiency:** The `OptimizedSTCM` provides the logical power of Tsetlin Machines with the implementation efficiency of a sparse linear layer.

### 4. Future Directions
- **Multi-Head STCM:** Implementing a Transformer-style multi-head mechanism where different STCM "heads" look at subspaces of features.
- **Int8/Bit Packing:** For inference, the ternary weights can be packed into 2-bit representations, potentially offering 16x memory reduction over Float32.

### 5. Recent Engineering Improvements
- **Optimized STCM default:** The capacity/product operators now use the linear
  mismatch projection path by default, matching the throughput of
  `OptimizedSTCM` without requiring a different class. Custom fuzzy operators
  automatically fall back to the legacy logic.
- **Clause memory for transformers:** `TMFeedForward` accepts shared
  `ClauseMemoryBank` instances, and `UnifiedTMTransformer` exposes
  `clause_memory_slots` to configure per-stage memories. Diagnostics now report
  memory peak/mean attention so you can verify whether slots are being used.
- **Tau/LF scheduler:** `trainers.TauLiteralScheduler` provides a single entry
  point for annealing STE hardness together with literal budgets, making staged
  curricula reproducible across MNIST/CIFAR baselines.

