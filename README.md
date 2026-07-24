# Fine-Tuning Vision Transformer (ViT) for Diabetic Retinopathy Severity Classification

> An end-to-end deep learning project focused on fine-tuning a pretrained Vision Transformer (ViT) using Hugging Face Transformers and PyTorch for multi-class diabetic retinopathy severity classification from retinal fundus images.

---

## 📌 Project Overview

Diabetic Retinopathy (DR) is one of the leading causes of preventable blindness among diabetic patients. Early and accurate detection of disease severity can significantly improve treatment outcomes.

This project aims to build a robust image classification pipeline by fine-tuning a pretrained **Vision Transformer (ViT)** model on retinal fundus images to classify the severity of diabetic retinopathy into five categories.

The project follows an industry-style workflow including dataset exploration, preprocessing, transfer learning, model fine-tuning, evaluation, and visualization.

---

## 🎯 Objectives

- Fine-tune a pretrained Vision Transformer (ViT-Base-Patch16-224)
- Perform multi-class retinal image classification
- Apply transfer learning using Hugging Face Transformers
- Evaluate model performance using multiple classification metrics
- Visualize predictions and confusion matrix
- Build a clean, modular, and reproducible deep learning pipeline

---

## 🩺 Classification Labels

| Label | Severity |
|--------|----------|
| 0 | No Diabetic Retinopathy |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative Diabetic Retinopathy |

---

## 📂 Dataset

**Dataset:** Diabetic Retinopathy 224×224 Gaussian Filtered

- Source: Kaggle
- Images are preprocessed, resized to **224×224**, and Gaussian filtered.
- Organized into five class folders for supervised image classification.

---

## 🛠 Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Git & GitHub

---

## 🧠 Model

- Vision Transformer (ViT)
- Pretrained on ImageNet
- Fine-tuned for 5-class diabetic retinopathy classification

Model:
- `google/vit-base-patch16-224`

---

## 📁 Project Structure

```text
fine-tuning-vit-diabetic-retinopathy/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│
├── models/
│
├── outputs/
│
├── README.md
├── requirements.txt
└── main.py
```

---

## 🚀 Project Workflow

- Dataset Exploration
- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Loading Pretrained ViT
- Transfer Learning
- Fine-Tuning
- Model Evaluation
- Confusion Matrix
- Prediction Visualization
- Performance Analysis

---

## 📊 Evaluation Metrics

The model will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 📈 Results

> 🚧 Currently under development.

The repository will be updated with:

- Training curves
- Validation curves
- Confusion Matrix
- Classification Report
- Sample Predictions
- Performance Analysis

---

## 📌 Future Improvements

- Data Augmentation
- Learning Rate Scheduling
- Hyperparameter Tuning
- Grad-CAM Visualization
- Model Comparison (CNN vs ViT)
- TensorBoard Integration

---

## 🎓 Learning Goals

This repository is being developed as a hands-on learning project to gain practical experience with:

- Vision Transformers (ViT)
- Transfer Learning
- Hugging Face Ecosystem
- Medical Image Classification
- Deep Learning Research Workflow
- Production-ready ML Project Structure

---

The project is actively being developed following an incremental, research-oriented workflow with detailed documentation and reproducible experiments.
