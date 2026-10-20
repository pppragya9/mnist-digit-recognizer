# Handwritten Digit Recognizer (MNIST)

A high-performance machine learning and deep learning project that classifies handwritten digits from the MNIST dataset using both a **Convolutional Neural Network (CNN)** in PyTorch (~99.2%+ accuracy) and a **Random Forest Baseline** in scikit-learn (~96.8% accuracy).

## Models & Accuracy Comparison

| Model | Architecture / Parameters | Test Accuracy | Inference Time per Image |
| :--- | :--- | :--- | :--- |
| **PyTorch CNN** (Default) | 2x Conv blocks (Conv2d, BatchNorm, ReLU, MaxPool, Dropout) + 2x FC layers | **~99.24%** | < 15 ms |
| **Random Forest** (Baseline) | 100 Trees (`n_estimators=100`, `max_features='sqrt'`) | **~96.75%** | ~50 ms |

## Features

- **High Accuracy (99.2%+)**: Deep CNN captures spatial structures, edges, and stroke geometry.
- **Model Persistence**: Automatically saves trained weights to `mnist_cnn.pth`. Instant inference on custom images without retraining.
- **Visual Evaluation**:
  - Confusion matrix heatmap (`confusion_matrix.png`).
  - True vs. predicted comparison grid on test samples (`sample_predictions.png`).
- **Custom Image Inference**: Supports predicting hand-drawn digits (dark ink on light paper or digital drawings) with softmax confidence scores.

## Installation

```bash
git clone https://github.com/pppragya9/mnist-digit-recognizer.git
cd mnist-digit-recognizer
pip install -r requirements.txt
```

## Usage

### 1. Train and Evaluate CNN
Trains the PyTorch CNN (5 epochs by default), saves the weights to `mnist_cnn.pth`, and generates evaluation plots:
```bash
python digit_recognizer.py
```

To customize epochs or force retraining:
```bash
python digit_recognizer.py --epochs 5 --retrain
```

### 2. Predict a Custom Digit
Loads the pre-trained `mnist_cnn.pth` and predicts the digit with confidence percentage in milliseconds:
```bash
python digit_recognizer.py my_digit.png
```

### 3. Run Random Forest Baseline
To compare with the classical Random Forest model:
```bash
python digit_recognizer.py --model rf
```

## Project Structure

```text
.
├── .gitignore
├── README.md
├── requirements.txt
├── digit_recognizer.py
├── mnist_cnn.pth           # Saved PyTorch CNN checkpoint (after training)
├── confusion_matrix.png     # Evaluation confusion matrix plot
└── sample_predictions.png   # Sample test predictions grid
```
