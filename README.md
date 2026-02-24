# Smoker Detection Model Using Deep Learning

## Introduction

Smoking detection from visual data is a binary image classification problem with applications in surveillance, public health monitoring, and behavioral research. This project focuses on building a deep learning-based model to classify whether a person is smoking or not from still images.

The system uses Transfer Learning with EfficientNet-B0 and is implemented using PyTorch. The project is designed for academic experimentation and research purposes.

---

## Dataset Description

The dataset consists of images categorized into two classes:

- Smokers
- Non-Smoker

The dataset is organized in the following structure:

DATA SET/
- Smokers/
- Non-Smoker/

The program automatically splits the dataset into:

- Training set (80%)
- Validation set (10%)
- Test set (10%)

This ensures proper model evaluation on unseen data.

---

## Data Preprocessing

The following preprocessing steps were applied:

- Resizing images to a fixed input size
- Normalization of pixel values
- Data augmentation techniques:
  - Random horizontal flipping
  - Random rotation

These techniques improve generalization and reduce overfitting.

---

## Model Architecture

The model uses EfficientNet-B0 pre-trained on ImageNet as the base feature extractor.

### Steps Followed

- Loaded EfficientNet-B0 with pre-trained weights
- Replaced the classification head for binary output
- Used a single output neuron for binary classification

### Model Configuration

- Loss Function: BCEWithLogitsLoss
- Optimizer: Adam
- Evaluation Metrics: Accuracy, Precision, Recall, F1-Score

The trained model weights are saved as:

smoker_classifier.pth

---

## Model Prediction

The model outputs a binary classification result:

- 0 → Non-Smoker
- 1 → Smoker

A prediction utility function allows classification of single images after training.

---

## Training and Evaluation

The training pipeline includes:

- Automatic dataset splitting
- Data loading using PyTorch DataLoader
- Model training with per-epoch loss and accuracy logging
- Evaluation on test data
- Classification report generation using scikit-learn

---

## Results and Impact

- Implemented transfer learning for binary image classification
- Built a complete training and evaluation pipeline
- Developed a reusable prediction function
- Designed for academic experimentation and research use

---

## Conclusion

This project demonstrates the use of deep learning and transfer learning for behavioral image classification. By leveraging EfficientNet-B0 and structured dataset handling, the model provides a clean and reproducible pipeline for binary classification tasks.

The implementation focuses on clarity, modularity, and academic usability rather than production deployment.
