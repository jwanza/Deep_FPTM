# Deep FPTM Documentation Index

This index condenses the large collection of Markdown documents that exist across the legacy Julia implementation, the new Python/STE stack, and earlier experiment archives.  The aim is to help you decide which originals you want to preserve when publishing to the new `Deep_FPTM` repository.

---

## 1. Core Concepts & Getting Started

| File(s) | Purpose | Keep? |
| --- | --- | --- |
| `README.md` | Research summary, datasets, Julia example commands, citation | ✅ canonical project overview |
| `python/fptm_ste/README.md` | PyTorch package layout, training utilities, CLI usage | ✅ include (renamed to `docs/python_usage.md` if desired) |
| `QUICK_START_GUIDE.md`, `FuzzyPatternTM-PyTorch/QUICK_START_ADVANCED.md`, `FuzzyPatternTM-PyTorch/QUICK_REFERENCE.md` | Step-by-step setup guides; most content overlaps | 🔄 fold into a single **Quick Start** document |
| `FuzzyPatternTM-PyTorch/README.md`, `FuzzyPatternTM-PyTorch/experiments/README.md` | Describes the pure PyTorch refactor, experiment structure | ✅ good background; shorten and move into `/docs/pytorch_overview.md` |

**Recommendation:** publish just two files for onboarding:

- `README.md` → tighten and link to the summary below.  
- `docs/QuickStart.md` → merge the various quick-start/checklist documents into one actionable guide (Julia + Python + datasets + environment tips).

---

## 2. Architecture & Implementation Notes

| Family of files | Highlights |
| --- | --- |
| `FuzzyPatternTM-PyTorch/DEEP_CODE_ANALYSIS.md`, `IMPLEMENTATION_COMPLETE.md`, `ANALYSIS_SUMMARY_README.md` | Detailed walkthroughs of module boundaries, clause evaluation, and STE gradients. These are useful reference notes but can be compressed into a single **Architecture Overview** section. |
| `IMPLEMENTATION_PLAN.md`, `IMPLEMENTATION_STATUS.md`, `STATUS_REPORT.md`, `FINAL_STATUS.md` | Project management history; optional if repo is geared toward users rather than documenting process. |
| `TM_FIX_SUMMARY.md`, `DEBUGGING_SUMMARY.md`, `DEBUGGING_SUCCESS_SUMMARY.md` | Fix logs for specific bug hunts; keep only if you want an engineering log. |

**Recommendation:** create `docs/Architecture.md` summarizing:

- Clause logic (fuzzy scoring vs binary).  
- STE training flow in both Julia (Flux) and Python (PyTorch).  
- Deep TM stack, Swin ensemble, TM-Transformer integration.  
- JSON cross-language bridge.

If needed, link to the raw investigation documents as appendices hosted in an `archive/` folder.

---

## 3. Optimisation & Performance Reports

| File | Content |
| --- | --- |
| `MASSIVE_SPEEDUP_RESULTS.md`, `OPTIMIZATIONS_APPLIED.md`, `OPTIMIZATION_QUICK_WINS.md`, `VECTORIZATION_SUCCESS.md`, `VECTORIZATION_FINAL.md`, `OPTIMIZATION_RESULTS_SUMMARY.md`, `FINAL_OPTIMIZATION_SUMMARY.md` | Benchmark breakdowns (pre/post), vectorisation strategies, memory optimisations.  Many share the same data table. |
| `EXECUTIVE_SUMMARY.md`, `FINAL_REPORT.md`, `COMPLETE_PROJECT_SUMMARY.md` | Narrative recap for stakeholders. |

**Recommendation:** combine into **one** concise report:

- `docs/Performance.md` with (a) baseline metrics, (b) key optimisations, (c) final numbers (throughput, latency, clause counts).  
- Move the step-by-step change logs to `docs/archive/perf_timeline.md` if you want to preserve the timeline.

---

## 4. Testing & Validation

| File | Focus |
| --- | --- |
| `TEST_SUMMARY.md`, `TESTING_DOCUMENTATION.md`, `FINAL_TEST_REPORT.md`, `TEST_EXECUTION_REPORT.md`, `TEST_FIXES_SUMMARY.md`, `TESTS_ADDED.md` | Unit/integration test coverage, flakes resolved, dataset-driven validations. |
| `JULIA_VS_PYTHON_ANALYSIS.md`, `COMPLETE_EQUIVALENCE_CHECK.md`, `IMPLEMENTATION_COMPARISON.md`, `COMPREHENSIVE_TEST_REPORT.md` | Cross-language equivalence experiments and KPIs. |
| `MNIST_TEST_RESULTS.md`, `TEST_AND_COMPARISON_REPORT.md` | Dataset-specific results. |

**Recommendation:** publish a single `docs/Validation.md` that includes:

1. How to run the automated test suites (Julia & Python).  
2. Summary table of MNIST/IMDb/CIFAR KPIs.  
3. Pointer to the JSON bridge + equivalence methodology.  
4. Optional appendix for per-model confusion matrices or logs.

---

## 5. Domain-Specific & Legacy Documents

| Location | Description |
| --- | --- |
| `dc_fptm/docs/maritime/*.md`, `dc_fptm/MARITIME_*` | A separate maritime analytics exploration (project plan, ranking, investment deck). Only include if the new repository still targets that domain. |
| `FINAL_SUMMARY.md`, `PROBLEM_SOLVED.md`, `INVESTIGATION_COMPLETE.md`, `FINAL_VALIDATION_REPORT.md`, `MASTER_EXECUTION_PLAN.md` | Retrospective or management reports.  Treat as history; move under `docs/archive/` or new GitHub wiki if needed. |

If “Deep_FPTM” is focused on the core TM research code, these files can stay in an `archive/` folder or be omitted from the new repo.  Provide a short note in the README indicating where to find historical planning documents if they remain public.

---

## 6. Proposed Minimal Markdown Set

To keep the GitHub project lightweight, the following Markdown files are recommended after consolidation:

1. `README.md` – brief overview, highlights, quick links.  
2. `docs/QuickStart.md` – setup + running instructions (Julia + Python).  
3. `docs/Architecture.md` – clause mechanics, STE training, deep variants, JSON bridge.  
4. `docs/Performance.md` – key optimization results (Julia + Python).  
5. `docs/Validation.md` – testing strategy, KPI tables, equivalence notes.  
6. `docs/DocumentationSummary.md` (this file) – index back to archived originals.  
7. Optionally: `docs/archive/*` for full historical reports you wish to preserve.

Everything else can either move under `docs/archive/` or be excluded entirely from the new Git history.

---

## 7. Next Steps

1. Move the selected original Markdown files into the new structure (or delete the redundant ones).  
2. Update the root README to link to the new `docs/` pages.  
3. Add an `.gitignore` that excludes local tooling (`venv/`, datasets, logs, `actions-runner/`, etc.) before committing.  
4. Stage and commit only the curated set when pushing to `Deep_FPTM`.

With this structure, the GitHub project presents a compact, navigable documentation set while keeping the detailed history available for reference when needed.

