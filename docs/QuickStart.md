# Deep FPTM Quick Start

This cheatsheet merges the scattered quick-start guides from the legacy Julia project and the new Python/STE stack.  Follow the steps that match your workflow.

---

## 1. Repository Layout

```
├── README.md                 ← research overview + key results
├── docs/                     ← consolidated documentation (architecture, validation, etc.)
│   ├── QuickStart.md         ← this guide
│   ├── Architecture.md*      ← clause mechanics, STE math (create next)
│   ├── Performance.md*       ← optimization highlights (create next)
│   ├── Validation.md*        ← testing & KPIs (create next)
│   └── DocumentationSummary.md
├── src/                      ← Julia core (fuzzy clauses, STE)
├── examples/                 ← Julia training scripts (MNIST, IMDb, etc.)
├── python/fptm_ste/          ← PyTorch package and training utilities
└── datasets/ (ignored)       ← optional local datasets
```

`*` These files are suggested in the documentation summary and can be populated as you curate the project.

---

## 2. Environment Setup

### 2.1 Python (PyTorch + STE)

```bash
# create a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# install core dependencies
pip install -r python/fptm_ste/requirements.txt

# expose the package for local imports
export PYTHONPATH="$(pwd)/python:${PYTHONPATH}"
```

Optional extras:

- `pip install -r python/fptm_ste/requirements-dev.txt` (if you publish one) for linters/tests.
- Set `TM_MNIST_EPOCHS`, `TM_MNIST_REPORT_EPOCH`, etc. to tweak the MNIST runner.

### 2.2 Julia (Flux + STE)

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

The project uses `common_venv` in other codebases; here we rely on Julia’s project manifest.  For GPU training, install CUDA.jl as needed.

---

## 3. Training Recipes

### 3.1 Python MNIST Baselines

```bash
source .venv/bin/activate
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
TM_MNIST_REPORT_EPOCH=1 TM_MNIST_EPOCHS=25 \
python python/fptm_ste/tests/run_mnist_equiv.py
```

Produces:

- Train/test accuracy per variant (STE TM, Deep TM, CNN Hybrid, TM Transformer).
- JSON exports under `/tmp/*.json` for cross-language checks.
- `/tmp/mnist_equiv_results.json` summarising KPIs.

### 3.2 Julia MNIST STE

```bash
julia --project=. examples/MNIST/mnist_ste.jl
```

This prints per-epoch train/validation accuracy (via the new metrics interface) and writes a compiled TM artefact at `/tmp/mnist_tm_ste_compiled.tm`.

### 3.3 Swin + Multi-Scale TM Ensemble (Python)

```bash
python python/fptm_ste/tests/run_cifar_ensemble.py \
  --epochs 10 --backbone swin_t --heads config/ms_heads.yaml
```

Key switches (pass as CLI args or modify your trainer):

- `--pretrained` to pull ImageNet weights.
- `--head-config` to select how TM heads map to Swin stages.
- `--log-uncertainty` to store agreement/disagreement statistics.

---

## 4. Testing & Validation

| Stack | Command | Notes |
| --- | --- | --- |
| Python | `pytest python/fptm_ste/tests -q` | Includes STE unit tests, Swin ensemble smoke tests, TM transformer checks. |
| Julia | `julia --project test/ste_smoke.jl` | Validates STE training, discretisation, JSON round-trip. |
| Cross-lang | Run MNIST trainer (Python) → export JSON → `julia scripts/predict_from_json_standalone.jl` | Compares predictions/accuracy between runtimes. |

Add GitHub Actions workflows later (CI for both languages).

---

## 5. Common Tasks

- **Discretise after training** (Python):

  ```python
  bundle = model.discretize(threshold=0.5)
  export_compiled_to_json(bundle, class_labels, "/tmp/my_tm.json", clauses_num=bundle["clauses_num"])
  ```

- **Load JSON in Julia**:

  ```julia
  using .FuzzyPatternTM: load_compiled_from_json, predict, accuracy
  tm = load_compiled_from_json("/tmp/my_tm.json")
  acc = accuracy(predict(tm, X_test), y_test)
  ```

- **Switch between soft/STE modes**:
  - Python defaults to soft (`use_ste_train=False`) during optimisation and hardens on evaluation/ export.
  - Julia exposes `ste` and `ste_eval` toggles in `train_ste!`.

---

## 6. Preparing for Git Commit

1. Copy or rename the handful of curated Markdown files you plan to publish (`README.md`, `docs/QuickStart.md`, etc.).  
2. Add a `.gitignore` to exclude `venv/`, `datasets/`, `__pycache__/`, `*.ipynb_checkpoints`, and large experiment logs.  
3. Stage only the curated source (`src/`, `examples/`, `python/fptm_ste/`, `docs/`, scripts).  
4. Commit with a message such as “Initial Deep FPTM release” and push to `https://github.com/jwanza/Deep_FPTM.git`.

---

## 7. What’s Next?

- Populate `docs/Architecture.md`, `docs/Performance.md`, and `docs/Validation.md` using the consolidation plan in `docs/DocumentationSummary.md`.  
- Trim or move legacy project-management documents to `docs/archive/`.  
- Add CI (PyTorch + Julia smoke tests) and packaging metadata (e.g., `pyproject.toml`) before tagging the first release.

With this quick-start checklist in place, new contributors can spin up either runtime and replicate the headline results in minutes.

