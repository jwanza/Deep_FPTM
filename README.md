# Fuzzy-Pattern Tsetlin Machine

A paradigm shift in the Tsetlin Machine family of algorithms.

## Abstract

The "*all-or-nothing*" clause evaluation strategy is a core mechanism in the Tsetlin Machine (TM) family of algorithms. In this approach, each clause—a logical pattern composed of binary literals mapped to input data—is disqualified from voting if even a single literal fails. Due to this strict requirement, standard TMs must employ thousands of clauses to achieve competitive accuracy. This paper introduces the **Fuzzy-Pattern Tsetlin Machine** (FPTM), a novel variant where clause evaluation is fuzzy rather than strict. If some literals in a clause fail, the remaining ones can still contribute to the overall vote with a proportionally reduced score. As a result, each clause effectively consists of sub-patterns that adapt individually to the input, enabling more flexible, efficient, and robust pattern matching. The proposed fuzzy mechanism significantly reduces the required number of clauses, memory footprint, and training time, while simultaneously improving accuracy.

On the IMDb dataset, FPTM achieves **90.15%** accuracy with **only one** clause per class, a **50×** reduction in clauses and memory over the Coalesced Tsetlin Machine. FPTM trains up to **316×** faster (**45 seconds** vs. **4 hours**) and fits within **50 KB**, enabling online learning on microcontrollers. Inference throughput reaches **34.5 million** predictions/second (51.4 GB/s). On Fashion-MNIST, accuracy reaches 92.18% (2 clauses), 93.19% (20 clauses) and **94.68%** (8000 clauses), a **∼400×** clause reduction compared to the Composite TM’s 93.00% (8000 clauses). On the Amazon Sales dataset with **20% noise**, FPTM achieves **85.22%** accuracy, significantly outperforming the Graph Tsetlin Machine (78.17%) and a Graph Convolutional Neural Network (66.23%).

## Changes compared to [Tsetlin.jl](https://github.com/BooBSD/Tsetlin.jl)

  - New fuzzy clause evaluation algorithm.
  - New hyperparameter `LF` that sets the number of literal misses allowed for the clause. The special case `LF = 1` corresponds to the same internal logic used in the [Tsetlin.jl](https://github.com/BooBSD/Tsetlin.jl) library.

The changes compared to [Tsetlin.jl](https://github.com/BooBSD/Tsetlin.jl) are located in the following functions: `check_clause()`, `feedback!()` and `train!()`.
Please, see the comments.

Here are the training results of the tiny **20-clause** model on the MNIST dataset:
<img width="698" alt="Fuzzy-Pattern Tsetlin Machine MNIST accuracy 98.56%" src="https://github.com/user-attachments/assets/05768a26-036a-40ce-b548-95925e96a01d">

## How to Run Examples

- Ensure that you have the latest version of the [Julia](https://julialang.org/downloads/) language installed.
- From the project root, instantiate the Julia environment to download all recorded dependencies:

```shell
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

  If you encounter an error about `Random` (or another standard library) not being installed, resolve the environment and retry:

```shell
julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
```
- Some examples require dataset preparation scripts written in [Python](https://www.python.org/downloads/). To install the necessary dependencies, run the following command:

```shell
pip install -r requirements.txt
```
In *all* Julia examples, we use `-t 32`, which specifies the use of `32` logical CPU cores.
Please adjust this parameter to match the actual number of logical cores available on your machine.

### IMDb Example (1 clause per class)

Prepare the IMDb dataset:

```shell
python examples/IMDb/prepare_dataset.py --max-ngram=4 --features=12800 --imdb-num-words=40000
```

Run the IMDb training and benchmarking example:

```shell
julia --project=. -O3 -t 32 examples/IMDb/imdb_minimal.jl
```

### IMDb Example (200 clauses per class)

Prepare the IMDb dataset:

```shell
python examples/IMDb/prepare_dataset.py --max-ngram=4 --features=65535 --imdb-num-words=70000
```

Run the IMDb training and benchmarking example:

```shell
julia --project=. -O3 -t 32 examples/IMDb/imdb_optimal.jl
```

### Noisy Amazon Sales Example

Prepare the noisy Amazon Sales dataset:

```shell
python examples/AmazonSales/prepare_dataset.py --dataset_noise_ratio=0.005
```

Run the Noisy Amazon Sales training example:

```shell
julia --project=. -O3 -t 32 examples/AmazonSales/amazon.jl
```

### Fashion-MNIST Example Using Convolutional Preprocessing

Run the Fashion-MNIST training example:

```shell
julia --project=. -O3 -t 32 examples/FashionMNIST/fmnist_conv.jl
```

### Fashion-MNIST Example Using Convolutional Preprocessing and Data Augmentation

To achieve maximum test accuracy, apply data augmentation when preparing the Fashion-MNIST dataset:

```shell
julia --project=. -O3 -t 32 examples/FashionMNIST/prepare_augmented_dataset.jl
```

Run the example that trains a large model on Fashion-MNIST:

```shell
julia --project=. -O3 -t 32 examples/FashionMNIST/fmnist_conv_augmented.jl
```

### CIFAR-10 Example Using Convolutional Preprocessing

Prepare the CIFAR-10 dataset:

```shell
julia --project=. -O3 -t 32 examples/CIFAR10/prepare_dataset.jl
```

Run the CIFAR-10 training example:

```shell
julia --project=. -O3 -t 32 examples/CIFAR10/cifar10_conv.jl
```

### MNIST Example

Run the MNIST training example:

```shell
julia --project=. -O3 -t 32 examples/MNIST/mnist.jl
```

Set `DATADEPS_ALWAYS_ACCEPT=true` (for example via `export DATADEPS_ALWAYS_ACCEPT=true`) if you prefer to auto-accept dataset download prompts when running in non-interactive environments.

To run the MNIST inference benchmark, please use the following command:

```shell
julia --project=. -O3 -t 32 examples/MNIST/mnist_benchmark_inference.jl
```

### Python Equivalence Runner

The PyTorch package in `python/fptm_ste/` now bundles a configurable MNIST/CIFAR battery that exercises the latest TM variants (straight-through, deep, Swin hybrid, and TM transformer) with gradient accumulation, STE annealing, EMA smoothing, and warm-up + cosine learning-rate schedules.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r python/fptm_ste/requirements.txt
python python/fptm_ste/tests/run_mnist_equiv.py
```

Optional environment flags:

| Variable | Purpose | Default |
|----------|---------|---------|
| `TM_MNIST_PRESET` | `baseline` (fast CNN/TM stack) or `advanced` (Swin, EMA, accumulation) | `baseline` |
| `TM_MNIST_EPOCHS` | Number of epochs per model | `25` (`baseline`), `50` (`advanced`) |
| `TM_MNIST_ACCUM`  | Gradient-accumulation steps | `1` (`baseline`), `4` (`advanced`) |
| `TM_MNIST_EMA`    | EMA decay factor | `0.0` (`baseline`), `0.999` (`advanced`) |
| `TM_MNIST_VARIANTS` | Comma-separated subset (`tm`, `deep_tm`, `hybrid`, `transformer`) | all |
| `TM_MNIST_REPORT_EPOCH` | Print per-epoch train accuracy | `0` (`baseline`), `1` (`advanced`) |
| `TM_MNIST_WARMUP` | Warm-up epochs before cosine decay | `0` (`baseline`), `5` (`advanced`) |

Examples:

```bash
# Only the deep TM variant with per-epoch logging
TM_MNIST_VARIANTS=deep_tm TM_MNIST_REPORT_EPOCH=1 python python/fptm_ste/tests/run_mnist_equiv.py

# Swin + TM hybrid with longer warm-up and small batch accumulation
TM_MNIST_PRESET=advanced TM_MNIST_VARIANTS=hybrid TM_MNIST_WARMUP=8 TM_MNIST_ACCUM=6 python python/fptm_ste/tests/run_mnist_equiv.py
```

Each run writes `/tmp/mnist_equiv_results.json` and, where applicable, JSON exports compatible with the Julia `JsonBridge`.  These artefacts allow you to verify cross-language equivalence or to load the compiled literals directly from Julia.

New Julia wrappers simply forward to the Python runner so you can stay inside the Julia workflow:

- `examples/MNIST/mnist_deep_tm.jl`
- `examples/MNIST/mnist_hybrid_swin.jl`
- `examples/MNIST/mnist_tm_transformer.jl`

They honour the same environment variables.  For example:

```bash
TM_MNIST_VARIANTS=hybrid julia --project=. examples/MNIST/mnist_hybrid_swin.jl
```

### Setun–Ternary Clause Machine (STCM)

The new `FuzzyPatternTM_STCM` module merges Setun-style ternary logic with the
fuzzy-pattern TM:

- One ternary state per feature per clause bank replaces the four literal
  matrices used by `FuzzyPatternTMFPTM`, cutting literal parameters in half.
- Clause operators are configurable (`capacity`/`product`), letting you choose
  between capacity−mismatch dynamics or fast product t-norms.
- Voting weights can stay continuous or switch to STE-based ternary logits for
  hardware-friendly exports.
- `DeepTMNetwork` accepts `layer_cls=FuzzyPatternTM_STCM` plus optional
  `layer_operator` / `layer_ternary_voting` hints to build multi-layer stacks.

See `docs/STCM.md` for a detailed architecture walkthrough, usage snippets, and
a summary of the accompanying unit + end-to-end test suite
(`python/tests/test_stcm_unit.py` and `python/tests/test_stcm_e2e.py`).

## Citation

If you use the Fuzzy-Pattern Tsetlin Machine in a scientific publication, please cite the following paper: [arXiv:2508.08350](https://arxiv.org/abs/2508.08350)

#### BibTeX:
```
@article{hnilov2025fptm,
    title={Fuzzy-Pattern Tsetlin Machine}, 
    author={Artem Hnilov},
    journal={arXiv preprint arXiv.2508.08350},
    year={2025},
    eprint={2508.08350},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2508.08350},
    doi = {10.48550/arXiv.2508.08350},
}
```

