# Medical Image Classification — Portfolio Project

This repository showcases a professional-grade, PyTorch-based medical image classification workflow rebuilt from the `integration` branch. It was developed and validated using Google Colab GPU training, with a focus on reproducibility, debugged production pipeline logic, and clear benchmark reporting.

The current `integration` README is the main project story. It explains how the code works, the data assumptions, the model family trade-offs, the green benchmarking strategy, and the exact experimental results captured in `integration/REPORT.md` and `integration/AUDIT_LOG.md`.

## What I built and why

- A full experimental pipeline supporting `AlexNet`, `VGG16`, `ResNet18`, and lightweight variants for the green-efficiency benchmark.
- A transfer-learning workflow for the small `organs` dataset using `scratch`, `feature_extraction`, and `fine_tune` modes.
- A robust dataset loader and preprocessing strategy that avoids train/validation/test leakage by using seeded splits and training-only normalization.
- A reproducible experiment configuration layer in `Code/config.json`, so the same setup can be rerun on Google Colab, local macOS, or another cloud GPU instance.
- A technical audit and recovery log that documents each defect, root cause, fix, and resulting benchmark improvement.

## Development environment and training platform

- Primary training environment: **Google Colab GPU runtime** (Tesla T4 / P100 style accelerator).
- Code language: **Python 3.x** with **PyTorch**, **NumPy**, **scikit-learn**, and **matplotlib**.
- Training was executed in short, reproducible runs using the central configuration in `Code/config.json`.
- The model and dataset files are organized so the repository can be cloned and a Colab notebook can launch the same CLI workflow with minimal changes.

## What changed in this integration branch

The audit log in `integration/AUDIT_LOG.md` contains the technical recovery story. Key corrections include:

- Standardized `Code/config.json` so datasets, models, hyperparameters, and task options are centralized.
- Fixed data slicing and random shuffling to prevent validation samples from leaking into training.
- Computed per-channel normalization statistics on training data only, then applied the same transform to validation and test.
- Corrected model constructors so `AlexNet`, `VGG16`, and `ResNet18` support arbitrary `in_channels` and `num_classes`.
- Restored the best validation checkpoint after training and prevented gradient accumulation by calling `optimizer.zero_grad()` each batch.
- Added lightweight variants and a shared runner to generate consistent CSV metrics, checkpoints, and history artifacts.

See `integration/AUDIT_LOG.md` for the full file-level defect log and exact commit references.

## Key benchmark outcomes from `integration/REPORT.md`

The `REPORT.md` file contains the complete run matrices for Task 1 (baseline), Task 2 (green benchmark), and Task 3 (scarce-data transfer). The top-line recommendations are:

- `cells`: best result from `Light_ResNet18` with **97.25% accuracy** and strong macro F1.
- `chest`: best result from baseline `ResNet18` with **89.90% accuracy**; lightweight variants reduced resource usage but lost accuracy.
- `lesions`: best result from `Light_ResNet18` with **76.16% accuracy** and improved efficiency.
- `orgs`: best result from `Light_ResNet18` with **92.15% accuracy** and the best overall green-efficiency trade-off.

### Task 2 green benchmark summary

| Dataset   | Best model       | Mode     | Accuracy | Macro F1 | Train time | Peak memory | Latency/sample |
| --------- | ---------------- | -------- | -------: | -------: | ---------: | ----------: | -------------: |
| `cells`   | `Light_ResNet18` | Light    |   97.25% |   97.02% |     685.7s |    659.7 MB |       0.826 ms |
| `chest`   | `ResNet18`       | Baseline |   89.90% |   88.57% |     327.0s |    833.9 MB |       1.020 ms |
| `lesions` | `Light_ResNet18` | Light    |   76.16% |   46.98% |     402.4s |    658.0 MB |       0.824 ms |
| `orgs`    | `Light_ResNet18` | Light    |   92.15% |   91.31% |     772.8s |    660.0 MB |       0.826 ms |

### Task 3 scarce-data transfer summary

The `organs` transfer experiment demonstrates that a pre-trained `Light_ResNet18` backbone plus partial fine-tuning is the strongest approach for a small 11-class target dataset.

| Mode               | Accuracy | Precision | Recall | Macro F1 | Trainable params | Train time | Peak memory |
| ------------------ | -------: | --------: | -----: | -------: | ---------------: | ---------: | ----------: |
| Scratch            |    58.0% |     52.3% |  55.0% |    50.8% |            4.18M |      23.7s |    652.7 MB |
| Feature extraction |    67.5% |     65.4% |  61.7% |    62.3% |             2.8k |      11.7s |    324.6 MB |
| Fine-tuning        |    70.0% |     67.0% |  62.8% |    63.5% |            2.27M |       9.3s |    353.2 MB |

> The takeaway: partial fine-tuning gives the best generalization on the small `organs` dataset while still benefiting from the source dataset features.

## How to reproduce the core experiments

1. Clone the repository and open the `integration` folder.
2. Ensure `data/*.pt` files are present in the repo root `data/` directory.
3. Use Google Colab or local Python with the following environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch numpy scikit-learn matplotlib
```

4. Run the task you want:

```bash
python3 Code/train.py --task task1
python3 Code/train.py --task task2
python3 Code/train.py --task task3
```

5. Inspect results in:

- `results/task1_test_metrics.csv`
- `results/task2_test_metrics.csv`
- `results/task3_test_metrics.csv`
- `results/model/`
- `results/history/`

## Data and preprocessing decisions

- Source files are expected as `data/cells.pt`, `data/chest.pt`, `data/lesions.pt`, `data/orgs.pt`, `data/organs.pt`.
- Each file must contain `train_images`, `train_labels`, `test_images`, `test_labels`.
- Data is split into training and validation with a seeded random split to ensure reproducibility and to prevent any ordered-slicing bias.
- Normalization uses per-channel mean/std from the training split only and applies the same transform to validation and test.
- Augmentations remain intentionally conservative to protect medical image semantics.

## Code structure and responsibilities

- `Code/config.json` — single source of truth for datasets, models, hyperparameters, tasks, and output paths.
- `Code/data.py` — dataset loading, normalization, augmentation, and data loader construction.
- `Code/models.py` — model definitions plus lightweight variants for efficient benchmarking.
- `Code/trainer.py` — mini-batch training, validation, early stopping, and best-state restoration.
- `Code/evaluate.py` — inference, metrics, and safe device handling.
- `Code/runner.py` — orchestration, experiment loop, checkpoint saving, and history logging.
- `Code/transfer.py` — transfer-learning experiment definitions and source checkpoint handling.
- `Code/train.py` — CLI entry point that selects the configured task and launches the runner.
- `Code/utils.py` — helper utilities for CSV output, plotting, path creation, and parameter counts.
- `tests/test_pipeline.py` — validation of config shape, synthetic data loading, model output shapes, and runner behavior.

## Senior developer perspective

This README is written to be understandable by a reviewer or teammate who needs to know:

- why the code was recovered and corrected,
- what the core experimental claims are,
- which dataset/model combinations were strongest,
- how to reproduce the work reliably,
- and where the exact numeric evidence lives (`integration/REPORT.md` and `integration/AUDIT_LOG.md`).

If you want, I can also add a short `integration/README.md` summary section specifically tailored for a GitHub portfolio landing page.
