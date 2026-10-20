# Handwritten Digit Recognizer (MNIST)

A machine learning project that trains a Random Forest Classifier on the MNIST handwritten digit dataset, evaluates performance with a confusion matrix and sample prediction grid, and classifies custom user-provided digit images.

## Features

- **MNIST Dataset**: Automatically fetches the 70,000 28x28 grayscale images via OpenML.
- **Random Forest Classifier**: Fast, parallelized multi-class digit recognition (`n_estimators=100`).
- **Visual Evaluation**:
  - Confusion matrix heatmap (`confusion_matrix.png`).
  - True vs. predicted comparison grid on test samples (`sample_predictions.png`).
- **Custom Image Inference**: Accepts any external image of a handwritten digit (dark digit on light background), inverts/resizes/normalizes it, and outputs the predicted digit.

## Installation

```bash
git clone https://github.com/pppragya9/mnist-digit-recognizer.git
cd mnist-digit-recognizer
pip install -r requirements.txt
```

## Usage

### 1. Train and Evaluate
Train the model, print test accuracy, and generate evaluation plots:
```bash
python digit_recognizer.py
```

### 2. Predict a Custom Digit
Pass the path to your own image file:
```bash
python digit_recognizer.py my_digit.png
```

## Project Structure

```text
.
├── .gitignore
├── README.md
├── requirements.txt
└── digit_recognizer.py
```
