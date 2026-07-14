# Medical Image Classification — Project Overview

This repository is a project of a robust medical image classification pipeline built with PyTorch. It demonstrates dataset engineering, model development (including lightweight variants), reproducible training, resource-aware benchmarking, and transfer-learning for scarce-data scenarios.

Summary: the project implements full training, evaluation, and reporting for multiple model families across several medical imaging datasets, with a focus on maintainability, reproducibility, and efficiency.

## What I built and why

- Reconstructed a modular training pipeline to reliably run `AlexNet`, `VGG16`, and `ResNet18` on several medical imaging datasets.
- Implemented lightweight variants (parameter-reduced versions) to explore efficiency vs accuracy trade-offs (the "green" initiative).
- Added a transfer-learning workflow to handle very small target datasets (`organs`) with `scratch`, `feature_extraction`, and `fine_tune` modes.
- Hardened data handling (seeded splits, training-only normalization), added configuration via `Code/config.json`, and included unit tests and result artifacts for reproducibility.

## High-level design

- Data: stored in `data/*.pt` files, each containing `train_images`, `train_labels`, `test_images`, `test_labels`.
- Models: parameterized PyTorch modules under `Code/models.py` with both baseline and lightweight variants.
- Training: `Code/trainer.py` implements batch training, validation, early stopping and best-state restoration.
- Orchestration: `Code/runner.py` builds datasets/loaders, trains models, evaluates test metrics and saves artifacts.
- Transfer learning: `Code/transfer.py` constructs `scratch`, `feature_extraction`, and `fine_tune` experiments using a source checkpoint.

## How the pipeline works (step-by-step)

1. Prepare data files in `data/` (see _Data format_ below).
2. Edit `Code/config.json` to pick dataset(s), model(s), hyperparameters, and output folders.
3. Run `python3 Code/train.py --task <task1|task2|task3>` to execute the chosen experiment set.
4. `Code/runner.py` will:
   - load the `.pt` dataset,
   - perform a seeded random train/validation split,
   - compute per-channel mean/std on the training split and apply z-score normalization to all splits,
   - instantiate the requested model and optimizer,
   - train with `CrossEntropyLoss` and Adam using `Code/trainer.py` (with `optimizer.zero_grad()` each batch),
   - save the best validation state and final test metrics to CSV under `results/`.

## Data format and preprocessing

- Expected files: `data/cells.pt`, `data/chest.pt`, `data/lesions.pt`, `data/orgs.pt`, `data/organs.pt`.
- Each file should contain 4 tensors: `train_images`, `train_labels`, `test_images`, `test_labels`.
- Train/validation split: a seeded random permutation ensures reproducible splits and prevents leakage between train and validation sets.
- Normalization: compute per-channel mean/std on the training split only, then apply the same statistics to validation and test — this prevents information leakage from validation/test into training.

## Augmentation: what and why (and why not aggressive)

- What: light geometric + photometric augmentation (small rotations, translations, scaling, brightness/contrast jitter) is used optionally in `Code/config.json`.
- Why light: medical images are sensitive to anatomical consistency. Aggressive augmentations (large rotations, heavy distortions or color shifts) can create unrealistic examples that harm clinical generalization. The chosen augmentations increase variability without breaking the semantics of anatomical structures.

## Model design and how classification is done

- Baselines: `AlexNet`, `VGG16`, `ResNet18` implementations accept flexible `in_channels` and `num_classes` and return logits for `CrossEntropyLoss`.
- Lightweight variants: created by reducing channel counts, decreasing fully-connected dimensions, and simplifying block repetition. The approach preserves the original architectural intent but reduces parameter counts (typical reductions: 50–75% parameters depending on family).
- Training: standard supervised classification training loop with mini-batch SGD via Adam, validation-based early stopping, and best-state restoration.

## Transfer learning and scarce-data strategy

- Modes implemented:
  - `scratch` — train from random init on the target dataset;
  - `feature_extraction` — freeze backbone, train a new classifier head on the target labels;
  - `fine_tune` — unfreeze final residual block(s) and classifier and adapt to the target.
- Rationale: with very few target images (`organs`), freezing most of the network preserves stable feature extractors learned on larger datasets and reduces overfitting risk. Fine-tuning a small subset of layers allows adaptation while keeping training cost low.

## Metrics and green benchmarking

- Metrics recorded per run: accuracy, macro precision, macro recall, macro F1, trainable/total parameter counts, training runtime, inference latency per sample, and peak training memory where available.
- Results are written to: `results/task1_test_metrics.csv`, `results/task2_test_metrics.csv`, `results/task3_test_metrics.csv` and checkpoints under `results/model/`.

## File-by-file guide (what each important file does)

- `Code/config.json` — central experiment configuration. Controls datasets, models, hyperparameters, augmentation settings, output paths, and task-specific options (task2, task3).
- `Code/data.py` — loads `.pt` dataset files, performs seeded train/validation splitting, computes training-only per-channel normalization, and builds PyTorch `Dataset`/`DataLoader` objects.
- `Code/models.py` — model definitions for `AlexNet`, `VGG16`, `ResNet18` and lightweight variants. Models accept `in_channels` and `num_classes`.
- `Code/trainer.py` — training loop, per-epoch training/validation steps, early stopping, `optimizer.zero_grad()`, gradient steps, and best-model tracking.
- `Code/evaluate.py` — test-time inference utilities and metric computation (handles device movement and uses `zero_division=0` to avoid undefined metrics).
- `Code/runner.py` — top-level orchestration for experiment runs: model construction, dataset loading, training invocation, metric collection, checkpoint saving, and history plotting.
- `Code/transfer.py` — helper to configure and run transfer-learning modes (`scratch`, `feature_extraction`, `fine_tune`) including source checkpoint loading.
- `Code/train.py` — CLI entrypoint that loads `Code/config.json`, parses `--task`, and drives the runner.
- `Code/utils.py` — I/O helpers for writing CSVs, creating output folders, plotting loss curves, and parameter counting.
- `tests/test_pipeline.py` — unit tests that validate config sections, model shapes, synthetic data loading, and pipeline-run helpers.
- `finaltaskmd/README.md`, `finaltaskmd/REPORT.md`, `finaltaskmd/AUDIT_LOG.md` — portfolio-focused copies of the project overview, consolidated report, and audit log for reference inside the submission folder.

## How I made the models lightweight (practical steps)

- Reduce the number of filters per convolutional stage (scale down 50–75% depending on family).
- Replace large fully-connected layers with smaller projection layers.
- Reduce intermediate channel widths and remove redundant feature-map expansions.
- Keep convolutional kernel sizes and basic block logic intact so transfer learning remains effective.

These changes keep the inductive bias of the original architectures while significantly reducing parameter counts and memory usage.

## Example commands

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch numpy scikit-learn matplotlib
```

Run a single task (task2 shows green benchmarking):

```bash
python3 Code/train.py --task task2
```

Run a transfer experiment for `organs` (task3):

```bash
python3 Code/train.py --task task3
```

## Reproducibility and development notes

- Seeds: runs use a fixed `SEED` in `Code/config.json` to keep train/validation splits and initialization deterministic.
- Logging: results CSVs and checkpoints are written under `results/` for each run; loss curves are stored under `results/history/`.
- Tests: run `python3 -m unittest discover -s tests` to validate the pipeline components locally.

## Why not aggressive augmentation or extreme normalization?

- Aggressive augmentation can distort medically relevant features (orientation, texture, relative size) and produce unrealistic samples. The augmentation policy focuses on conservative geometric and photometric changes.
- Normalization is dataset-aware and computed on training data only to avoid leaking information from validation or test sets into training statistics; per-channel normalization preserves color-space structure for multi-channel medical images.

## Next steps I recommend

- Add class-stratified cross-validation runs to estimate variance across seeds.
- Expand domain-specific augmentation with clinician input for more realistic synthetic variations.
- Add a short notebooks/ dashboard that reads `results/*.csv` for quick interactive inspection when presenting in a portfolio.

---

If you want, I can now commit these README updates and prepare a standalone repo with these files ready for your GitHub push.
