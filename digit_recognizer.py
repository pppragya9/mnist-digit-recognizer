"""
Handwritten Digit Recognizer (MNIST)
------------------------------------
Trains a Random Forest on MNIST, reports accuracy, saves a confusion
matrix and sample predictions, and can predict your own digit image.

Usage:
    python digit_recognizer.py                  # train + evaluate
    python digit_recognizer.py my_digit.png     # train, then predict your image
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------- 1. DATA
def load_data():
    """Download MNIST (70,000 images, 28x28) and scale pixels to 0-1."""
    print("Loading MNIST...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X / 255.0
    y = y.astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ---------------------------------------------------------------- 2. MODEL
def train_model(X_train, y_train):
    """Train a Random Forest classifier."""
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    return model


# ------------------------------------------------------------ 3. EVALUATION
def evaluate(model, X_test, y_test):
    """Print test accuracy and return predictions."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    print(f"Test accuracy: {acc:.2f}%")
    return preds


# ------------------------------------------------------------- 4. PLOTS
def save_confusion_matrix(y_test, preds, path="confusion_matrix.png"):
    """Show which digits get mixed up and save the plot."""
    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def save_sample_predictions(X_test, y_test, preds, path="sample_predictions.png"):
    """Save a grid of 10 test images with true vs predicted labels."""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for ax, img, true, pred in zip(axes.ravel(), X_test, y_test, preds):
        ax.imshow(img.reshape(28, 28), cmap="gray")
        ax.set_title(f"True {true} / Pred {pred}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ------------------------------------------------- 5. PREDICT YOUR CARD/IMAGE
def predict_image(model, path):
    """Predict a digit from your own image (dark digit on light paper)."""
    from PIL import Image, ImageOps

    img = Image.open(path).convert("L")
    img = ImageOps.invert(img)  # MNIST is a white digit on black
    img = img.resize((28, 28))
    arr = np.array(img).reshape(1, -1) / 255.0
    print(f"Predicted digit for {path}: {model.predict(arr)[0]}")


# ---------------------------------------------------------------- MAIN
def main():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    preds = evaluate(model, X_test, y_test)

    save_confusion_matrix(y_test, preds)
    save_sample_predictions(X_test, y_test, preds)

    if len(sys.argv) > 1:
        predict_image(model, sys.argv[1])


if __name__ == "__main__":
    main()
