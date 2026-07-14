# Consolidated Benchmark Report

## Executive Summary

The reconstructed pipeline trains and evaluates the recovered convolutional classifiers through a shared configuration file. The final artifacts cover the corrected Task 1 baseline matrix, the Task 2 Green Initiative matrix, and the Task 3 scarce-data transfer experiment.

The final result artifacts are stored in:

```text
results/task1_test_metrics.csv
results/task2_test_metrics.csv
results/task3_test_metrics.csv
```

All final runs use 20 epochs with early stopping enabled. The CSV files include classification metrics plus green-efficiency measurements: parameter count, training runtime, peak memory, inference runtime, and inference latency per sample.

## Assignment Accuracy Targets

| Dataset   | Minimal accuracy |
| --------- | ---------------: |
| `cells`   |              90% |
| `chest`   |              87% |
| `lesions` |              67% |
| `orgs`    |              83% |

## Current Validated Benchmark Results

These results are taken from the completed 20-epoch Task 2 CSV in `results/task2_test_metrics.csv`. Task 2 contains both the baseline models and lightweight variants, so this table selects the best final model for each assignment dataset from the complete benchmark matrix.

| Dataset   | Best model       | Mode     | Accuracy | Precision | Recall | Macro F1 | Target | Status |
| --------- | ---------------- | -------- | -------: | --------: | -----: | -------: | -----: | ------ |
| `cells`   | `Light_ResNet18` | Light    |   97.25% |    97.18% | 96.93% |   97.02% | 90.00% | Pass   |
| `chest`   | `ResNet18`       | Baseline |   89.90% |    92.44% | 86.79% |   88.57% | 87.00% | Pass   |
| `lesions` | `Light_ResNet18` | Light    |   76.16% |    54.22% | 44.12% |   46.98% | 67.00% | Pass   |
| `orgs`    | `Light_ResNet18` | Light    |   92.15% |    91.51% | 91.34% |   91.31% | 83.00% | Pass   |

## Task 1 Baseline Matrix

Task 1 evaluates the corrected baseline `AlexNet`, `VGG16`, and `ResNet18` implementations across the four required datasets.

| Dataset   | Model      | Epochs | Parameters | Train time | Train memory | Accuracy | Precision | Recall | Macro F1 | Latency/sample |
| --------- | ---------- | -----: | ---------: | ---------: | -----------: | -------: | --------: | -----: | -------: | -------------: |
| `cells`   | `AlexNet`  |     20 |  5,693,544 |      60.3s |     186.6 MB |   94.18% |    94.93% | 92.67% |   93.59% |       0.116 ms |
| `cells`   | `VGG16`    |     20 | 12,631,624 |     380.2s |     607.0 MB |   97.16% |    97.21% | 96.66% |   96.90% |       0.491 ms |
| `cells`   | `ResNet18` |     20 | 11,172,936 |     635.5s |     837.4 MB |   96.73% |    96.39% | 96.64% |   96.46% |       1.004 ms |
| `chest`   | `AlexNet`  |     20 |  5,682,690 |      28.5s |     183.5 MB |   86.70% |    91.23% | 82.26% |   84.41% |       0.105 ms |
| `chest`   | `VGG16`    |     20 | 12,627,394 |     103.2s |     608.8 MB |   89.42% |    91.58% | 86.41% |   88.07% |       0.490 ms |
| `chest`   | `ResNet18` |     20 | 11,168,706 |     327.0s |     833.9 MB |   89.90% |    92.44% | 86.79% |   88.57% |       1.020 ms |
| `lesions` | `AlexNet`  |     20 |  5,692,519 |      48.6s |     185.9 MB |   75.46% |    53.26% | 47.11% |   48.07% |       0.114 ms |
| `lesions` | `VGG16`    |     20 | 12,631,111 |     225.9s |     609.6 MB |   72.92% |    43.98% | 38.49% |   38.13% |       0.488 ms |
| `lesions` | `ResNet18` |     20 | 11,172,423 |     496.8s |     835.1 MB |   76.11% |    60.98% | 48.32% |   50.84% |       1.016 ms |
| `orgs`    | `AlexNet`  |     20 |  5,691,915 |      87.5s |     185.7 MB |   89.51% |    88.58% | 88.60% |   88.34% |       0.094 ms |
| `orgs`    | `VGG16`    |     20 | 12,632,011 |     433.3s |     610.4 MB |   90.62% |    89.70% | 89.89% |   89.57% |       0.483 ms |
| `orgs`    | `ResNet18` |     20 | 11,173,323 |     958.1s |     838.0 MB |   91.98% |    90.68% | 90.92% |   90.68% |       0.999 ms |

## Full Task 2 Benchmark Matrix

Task 2 compares each baseline architecture with its lightweight counterpart. Baseline rows are included here as the reference point for the Green Initiative comparison.

| Dataset   | Model            | Mode     | Epochs | Parameters | Train time | Train memory | Accuracy | Precision | Recall | Macro F1 | Latency/sample |
| --------- | ---------------- | -------- | -----: | ---------: | ---------: | -----------: | -------: | --------: | -----: | -------: | -------------: |
| `cells`   | `AlexNet`        | Baseline |     20 |  5,693,544 |      60.3s |     186.6 MB |   94.18% |    94.93% | 92.67% |   93.59% |       0.116 ms |
| `cells`   | `Light_AlexNet`  | Light    |     20 |  1,522,472 |      67.1s |     115.2 MB |   96.17% |    95.65% | 96.40% |   95.93% |       0.129 ms |
| `cells`   | `VGG16`          | Baseline |     20 | 12,631,624 |     380.2s |     607.0 MB |   97.16% |    97.21% | 96.66% |   96.90% |       0.491 ms |
| `cells`   | `Light_VGG16`    | Light    |     20 |  4,447,752 |     287.9s |     409.8 MB |   95.73% |    96.48% | 94.94% |   95.62% |       0.375 ms |
| `cells`   | `ResNet18`       | Baseline |     20 | 11,172,936 |     635.5s |     837.4 MB |   96.73% |    96.39% | 96.64% |   96.46% |       1.004 ms |
| `cells`   | `Light_ResNet18` | Light    |     20 |  4,184,008 |     685.7s |     659.7 MB |   97.25% |    97.18% | 96.93% |   97.02% |       0.826 ms |
| `chest`   | `AlexNet`        | Baseline |     20 |  5,682,690 |      28.5s |     183.5 MB |   86.70% |    91.23% | 82.26% |   84.41% |       0.105 ms |
| `chest`   | `Light_AlexNet`  | Light    |     20 |  1,517,378 |      20.9s |     112.7 MB |   84.29% |    89.37% | 79.23% |   81.29% |       0.123 ms |
| `chest`   | `VGG16`          | Baseline |     20 | 12,627,394 |     103.2s |     608.8 MB |   89.42% |    91.58% | 86.41% |   88.07% |       0.490 ms |
| `chest`   | `Light_VGG16`    | Light    |     20 |  4,445,826 |     103.9s |     407.0 MB |   86.38% |    90.29% | 82.09% |   84.13% |       0.373 ms |
| `chest`   | `ResNet18`       | Baseline |     20 | 11,168,706 |     327.0s |     833.9 MB |   89.90% |    92.44% | 86.79% |   88.57% |       1.020 ms |
| `chest`   | `Light_ResNet18` | Light    |     20 |  4,181,314 |     251.1s |     653.8 MB |   83.65% |    89.32% | 78.29% |   80.35% |       0.819 ms |
| `lesions` | `AlexNet`        | Baseline |     20 |  5,692,519 |      48.6s |     185.9 MB |   75.46% |    53.26% | 47.11% |   48.07% |       0.114 ms |
| `lesions` | `Light_AlexNet`  | Light    |     20 |  1,522,407 |      41.1s |     115.2 MB |   75.21% |    49.51% | 42.57% |   45.23% |       0.113 ms |
| `lesions` | `VGG16`          | Baseline |     20 | 12,631,111 |     225.9s |     609.6 MB |   72.92% |    43.98% | 38.49% |   38.13% |       0.488 ms |
| `lesions` | `Light_VGG16`    | Light    |     20 |  4,447,623 |     169.2s |     409.2 MB |   71.92% |    37.78% | 30.37% |   29.99% |       0.382 ms |
| `lesions` | `ResNet18`       | Baseline |     20 | 11,172,423 |     496.8s |     835.1 MB |   76.11% |    60.98% | 48.32% |   50.84% |       1.016 ms |
| `lesions` | `Light_ResNet18` | Light    |     20 |  4,183,751 |     402.4s |     658.0 MB |   76.16% |    54.22% | 44.12% |   46.98% |       0.824 ms |
| `orgs`    | `AlexNet`        | Baseline |     20 |  5,691,915 |      87.5s |     185.7 MB |   89.51% |    88.58% | 88.60% |   88.34% |       0.094 ms |
| `orgs`    | `Light_AlexNet`  | Light    |     20 |  1,517,963 |      72.3s |     114.2 MB |   90.01% |    88.66% | 89.03% |   88.72% |       0.088 ms |
| `orgs`    | `VGG16`          | Baseline |     20 | 12,632,011 |     433.3s |     610.4 MB |   90.62% |    89.70% | 89.89% |   89.57% |       0.483 ms |
| `orgs`    | `Light_VGG16`    | Light    |     20 |  4,446,987 |     321.8s |     406.5 MB |   91.35% |    90.63% | 90.18% |   90.23% |       0.368 ms |
| `orgs`    | `ResNet18`       | Baseline |     20 | 11,173,323 |     958.1s |     838.0 MB |   91.98% |    90.68% | 90.92% |   90.68% |       0.999 ms |
| `orgs`    | `Light_ResNet18` | Light    |     20 |  4,183,627 |     772.8s |     660.0 MB |   92.15% |    91.51% | 91.34% |   91.31% |       0.826 ms |

## Methodology

The current pipeline sets Python/PyTorch random seeds, uses a seeded random train/validation split, computes z-score normalization statistics from the training split only, trains with `CrossEntropyLoss` and Adam, restores the best validation-loss model after early stopping, and logs macro-averaged precision/recall/F1 plus runtime, memory, and latency.

Macro-averaged metrics are appropriate because the datasets are multi-class and may have class imbalance. Macro averaging gives each class equal weight instead of allowing large classes to dominate the score.

## Architecture Recommendations

| Dataset   | Recommendation   | Reason                                                                                                                                      |
| --------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `cells`   | `Light_ResNet18` | Highest accuracy and macro F1 while using fewer parameters than baseline `ResNet18`.                                                        |
| `chest`   | `ResNet18`       | Best accuracy and macro F1; the lightweight version loses too much accuracy on this dataset.                                                |
| `lesions` | `Light_ResNet18` | Slightly highest accuracy and lower memory/latency than baseline `ResNet18`; macro F1 remains weak, so class imbalance should be discussed. |
| `orgs`    | `Light_ResNet18` | Highest accuracy and macro F1 with lower memory and latency than baseline `ResNet18`.                                                       |

## Green Initiative Analysis

Task 2 adds lightweight versions of the restored model families:

- `Light_AlexNet`
- `Light_VGG16`
- `Light_ResNet18`

| Model family | Baseline parameters | Lightweight parameters | Parameter reduction |
| ------------ | ------------------: | ---------------------: | ------------------: |
| AlexNet      |           5,690,167 |              1,520,055 |              73.29% |
| VGG16        |          12,630,535 |              4,447,047 |              64.79% |
| ResNet18     |          11,171,847 |              4,183,175 |              62.56% |

Parameter counts vary slightly by dataset because the classifier output size changes with the number of classes. The table reports the mean parameter count across the four datasets.

The next table summarizes average cost changes from each baseline family to its lightweight counterpart across the four assignment datasets.

| Model family | Parameter reduction | Train-memory reduction | Latency reduction | Training-time change | Accuracy change |
| ------------ | ------------------: | ---------------------: | ----------------: | -------------------: | --------------: |
| AlexNet      |              73.29% |                 38.34% |            -5.78% |              -10.49% |        -0.04 pp |
| VGG16        |              64.79% |                 32.98% |            23.23% |              -22.74% |        -1.19 pp |
| ResNet18     |              62.56% |                 21.32% |            18.44% |              -12.63% |        -1.38 pp |

The lightweight models substantially reduce parameter count and memory. `Light_VGG16` and `Light_ResNet18` also reduce inference latency on average. `Light_AlexNet` reduces parameters and memory but has slightly worse average latency in these runs, so its main benefit is model-size reduction rather than faster inference.

The completed benchmark supports `Light_ResNet18` as the best overall green recommendation for `cells`, `lesions`, and `orgs`. It keeps the residual architecture's accuracy advantage while reducing parameter count, memory footprint, and inference latency relative to baseline `ResNet18`. For `chest`, baseline `ResNet18` should be kept because it is the strongest result above the 87% target.

## Task 3 — Organs Classification Knowledge Transfer Adaptation

The new `organs` dataset contains 11 classes but is much smaller than other existing dataset. Therefore, Task 3 evaluates whether transfer learning from larger and most alike`orgs` dataset can provide a robust model for the new organs classification task.

`Light_ResNet18_orgs` was selected as the pre-trained source model because it achieved a strong balance between accuracy and efficiency on the orgs dataset in Task 2.. Also, the`orgs`dataset contains 15,367 training images and is visually the most similar dataset to small `organs` dataset which contains only 500 images.

### Experimental Setup

All experiments used a validation split of 0.20, a maximum of 30 epochs, a learning rate of 0.001, and a dropout rate of 0.5. A fixed random seed of 42 was used for reproducibility. The following training modes were compared:

| Mode               | Description                                                                                   |
| ------------------ | --------------------------------------------------------------------------------------------- |
| Scratch            | Random initialization; trained only on the small `organs` dataset                             |
| Feature extraction | Pre-trained `orgs` backbone frozen; only the new 11-class classifier trained                  |
| Fine-tuning        | Pre-trained model partially fine-tuned; the classifier and final residual block were unfrozen |

The result is reported below **without data augmentation**.

All approaches exceeded the required minimum test accuracy of 40%. Transfer learning improved performance compared with scratch training, showing that features learned from the larger `orgs` dataset were transferable to the new `organs` dataset.

### Classification Performance

| Mode               | Test Accuracy | Macro Precision | Macro Recall |  Macro F1 |
| ------------------ | ------------: | --------------: | -----------: | --------: |
| Scratch            |         58.0% |           52.3% |        55.0% |     50.8% |
| Feature extraction |         67.5% |           65.4% |        61.7% |     62.3% |
| Fine-tuning        |     **70.0%** |       **67.0%** |    **62.8%** | **63.5%** |

### Computational Efficiency

| Mode               | Trainable Parameters | Training Time | Peak Training Memory | Inference Latency / Sample |
| ------------------ | -------------------: | ------------: | -------------------: | -------------------------: |
| Scratch            |            4,183,627 |        23.7 s |             652.7 MB |                   0.845 ms |
| Feature extraction |                2,827 |          11.7 |             324.6 MB |                   0.761 ms |
| Fine-tuning        |            2,266,379 |         9.3 s |             353.2 MB |                   0.766 ms |

The scratch model achieved the lowest test accuracy (58.0%). It also required the most training time, peak GPU memory, and inference time. In contrast, both transfer-learning approaches achieved better classification performance with lower computational cost. Fine-tuning had the shortest training time because training was ended by early stopping at epoch 13.

### Effect of Data Augmentation

To further investigate whether the synthetic image made from data augmentation could increase the performance on the limited target dataset. The augmentation was applied random affine transformations, including rotations of up to 10 degrees, translations of up to 5%, and scaling between 0.9 and 1.1. In addition, brightness and contrast were randomly adjusted by up to 20%.

| Mode               | Without Augmentation | With Augmentation | Effect             |
| ------------------ | -------------------: | ----------------: | ------------------ |
| Scratch            |                58.0% |         **66.5%** | Improved           |
| Feature extraction |            **67.5%** |             65.0% | Decreased          |
| Fine-tuning        |                70.0% |             70.0% | No accuracy change |

With data augmentation, scratch model substantially improved test accuracy from (58.0%) to (66.5%). Without augmentation, the best validation performance was reached at epoch 13, whereas augmentation delayed the best epoch to 22. This is because the model started memorizing the training dataset earlier with non-augmented dataset, while augmentation made the learning task more challenging and allowed the model to train longer.

In contrast, feature extraction performance decreased from (67.5%) to (65.0%). Since the pre-trained backbone was frozen and only the final classifier was unfrozen,the model had limited ability to adapt to the augmented images. Therefore, updating only the classifier may not have been enough for the model to handle the augmented images effectively.

For fine-tuning, although augmentation helped reduce over-fitting during training, it did not change test accuracy, which remained at 70.0%. Because only the last residual block and classifier were trainable, the model could adapt more effectively to the augmented data than the feature-extraction model. However, the earlier frozen layers may have limited the model’s ability to adapt to the changes introduced by augmentation.

### Task 3 Recommendation

For the current small `organs` dataset, we recommend partial fine-tuning of Light_ResNet18 as it achieved the highest test accuracy while preserving trainable parameters, training time, and less peak memory than scratch model.

### Task 3 Limitations

The main limitation is the small size of the `organs` dataset. With only around 500 images across 11 classes, the number of training samples per class is limited. For classes with similar visual patterns, the model found difficult to learn reliable differences between them.

To address this limitation, we applied data augmentation to increase the variability of the training images. However, the augmentation experiment has limitations. Since we lack medical domain knowledge, we could not determine which data transformations would not negatively affect the model to learn important characteristics of each classes. Therefore, we restricted the augmentation to general image transformations, such as changes in brightness, contrast, rotation, and scale, which may not reflect how organ images actually vary in real `organs` data. Future work should evaluate the models on a larger dataset, apply cross-validation to obtain more reliable performance estimates and investigate augmentation settings with medical expert knowledge.

## Current Limitations

- The results come from one run per configuration, so random-seed variance is not measured.
- The `lesions` dataset has weak macro F1 despite passing the accuracy target.
- Memory values are runtime-dependent and may differ slightly on another GPU.
