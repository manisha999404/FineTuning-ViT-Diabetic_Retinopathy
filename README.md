# Diabetic Retinopathy Detection using Vision Transformer

A Vision Transformer (ViT) based image classification project for detecting and grading diabetic retinopathy from retinal fundus images.

The project focuses not only on achieving good classification accuracy, but also on analyzing model behavior under class imbalance and evaluating how different fine-tuning strategies affect performance.

---

## Problem Statement

Diabetic retinopathy (DR) is a diabetes-related eye disease that can lead to vision loss if not detected early.

The objective of this project is to automatically classify retinal fundus images into five stages of diabetic retinopathy:

- No DR
- Mild
- Moderate
- Severe
- Proliferative DR

This is formulated as a five-class image classification problem.

---

## Approach

The project uses a pretrained **Vision Transformer (ViT-Base/16)** and fine-tunes it on retinal fundus images.

### Overall Pipeline

```text
Retinal Fundus Image
        ↓
Image Preprocessing
        ↓
224 × 224 Image
        ↓
ViT Patch Embedding
        ↓
Transformer Encoder
        ↓
Classification Head
        ↓
5 DR Severity Classes
```

The model was initialized with pretrained ImageNet weights and fine-tuned for the diabetic retinopathy classification task.

---

## Dataset

The project uses a diabetic retinopathy dataset containing retinal fundus images belonging to five severity levels.

### Class Distribution

| Class | Images |
|---|---:|
| No DR | 1,805 |
| Mild | 370 |
| Moderate | 999 |
| Severe | 193 |
| Proliferative DR | 295 |
| **Total** | **3,662** |

The dataset is highly imbalanced, with the **No DR** and **Moderate** classes containing significantly more samples than the **Severe** and **Proliferative DR** classes.

A stratified train-validation split was used to preserve the class distribution.

### Data Split

| Split | Samples |
|---|---:|
| Training | 2,929 |
| Validation | 733 |

---

## Model

### Vision Transformer — ViT-Base/16

The pretrained ViT model uses:

- Patch size: `16 × 16`
- Input resolution: `224 × 224`
- Hidden dimension: `768`
- Transformer layers: `12`
- Classification classes: `5`

The original ImageNet classification head was replaced with a five-class classification head.

---

## Experimental Design

Several controlled experiments were performed to investigate the effect of different fine-tuning strategies.

### Experiment 1 — Baseline ViT

Standard ViT fine-tuning using:

- AdamW optimizer
- Learning rate: `2e-5`
- Weight decay: `0.01`
- Standard Cross-Entropy Loss
- Batch size: `16`
- 3 epochs

### Experiment 2 — Class-Weighted Loss

To address class imbalance, class-specific weights were incorporated into Cross-Entropy Loss.

The objective was to give greater importance to underrepresented diabetic retinopathy severity classes.

### Experiment 3 — Data Augmentation

Mild image augmentation was introduced during training:

- Random horizontal flip
- Random rotation (`±10°`)

The validation set was kept unaugmented.

### Experiment 4 — Learning Rate Sensitivity

The learning rate was reduced from:

```text
2e-5 → 1e-5
```

All other experimental settings were kept unchanged to isolate the effect of learning rate.

---

## Results

### Validation Performance

| Experiment | Main Change | Accuracy | Macro-F1 |
|---|---|---:|---:|
| **Experiment 1 — Baseline** | Standard ViT | **80.22%** | **0.6289** |
| Experiment 2 — Class Weighting | Weighted Cross-Entropy | 73.40% | 0.5966 |
| Experiment 3 — Augmentation | Flip + Rotation | 78.58% | 0.5176 |
| Experiment 4 — Learning Rate | LR = `1e-5` | 76.67% | 0.6004 |

> **Note:** These are validation results. Final evaluation and additional analysis will be performed on the selected model.

---

## Key Findings

### 1. Baseline ViT performed best overall

The baseline configuration achieved the highest validation accuracy and Macro-F1:

- **Accuracy:** 80.22%
- **Macro-F1:** 0.6289

Therefore, it is currently the best-performing configuration among the tested experiments.

### 2. Class weighting introduced a performance trade-off

Class-weighted Cross-Entropy was introduced to address the dataset imbalance.

Although class weighting improved recall for some minority classes, overall validation performance decreased.

This demonstrates that increasing the importance of minority classes can create a trade-off between minority-class sensitivity and overall performance.

### 3. Simple augmentation did not improve performance

Horizontal flipping and small rotations did not outperform the baseline.

The final Macro-F1 decreased to **0.5176**.

This suggests that the selected augmentation strategy was not beneficial under the current model and dataset configuration.

### 4. Learning rate affected fine-tuning performance

Reducing the learning rate from `2e-5` to `1e-5` resulted in a Macro-F1 of **0.6004**, lower than the baseline Macro-F1 of **0.6289**.

Under the tested conditions, `2e-5` was therefore the better learning rate.

---

## Evaluation Metrics

Because the dataset is imbalanced, accuracy alone is not sufficient for evaluating the model.

The project therefore uses multiple evaluation metrics.

### Accuracy

Measures the overall proportion of correctly classified images.

### Macro-F1

Computes the F1-score independently for each class and then takes the unweighted average.

Macro-F1 is particularly important for this dataset because it gives minority classes equal importance instead of allowing the majority class to dominate the metric.

### Confusion Matrix

A confusion matrix is used to analyze which diabetic retinopathy severity levels are most frequently confused by the model.

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Vision Transformer (ViT)
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Google Colab
- Git
- GitHub

---

## Project Structure

```text
FineTuning-ViT-Diabetic_Retinopathy/
│
├── notebooks/
│   ├── 01_vit_setup.ipynb
│   ├── 02_dataset_exploration.ipynb
│   └── baseline_training.ipynb
│
├── src/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

Large datasets, trained model weights, virtual environments, and experiment outputs are intentionally excluded from the Git repository.

---

## Future Work

Possible extensions include:

- More targeted medical-image augmentation
- Improved handling of class imbalance
- Learning-rate scheduling
- Longer fine-tuning
- Comparison with CNN-based architectures
- Test-set evaluation
- Detailed per-class error analysis
- Attention visualization
- Model interpretability
- Grad-CAM and attention-based visualization
- Hyperparameter optimization

---

## Current Status

### Completed

- [x] Dataset exploration
- [x] Class distribution analysis
- [x] Stratified train-validation split
- [x] ViT model setup
- [x] Baseline fine-tuning
- [x] Class-weighted loss experiment
- [x] Data augmentation experiment
- [x] Learning-rate sensitivity experiment
- [x] Comparative validation analysis

### In Progress

- [ ] Final evaluation of the best model
- [ ] Confusion matrix and per-class analysis
- [ ] Result visualization
- [ ] Final model documentation
- [ ] Model interpretability analysis

---

## Research Takeaway

The experiments show that increasing model complexity or adding common training techniques does not necessarily improve performance.

Under the tested configurations, the **baseline ViT with a learning rate of `2e-5` achieved the strongest validation performance**.

The experiments also highlight the importance of evaluating models using class-aware metrics such as **Macro-F1**, particularly when working with imbalanced medical image datasets.

---

## Disclaimer

This project is intended for **research and educational purposes only** and is not a medical diagnostic system.
