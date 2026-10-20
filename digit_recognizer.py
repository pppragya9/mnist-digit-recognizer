"""
Handwritten Digit Recognizer (MNIST)
------------------------------------
Trains a Convolutional Neural Network (CNN) in PyTorch (or Random Forest in scikit-learn)
on MNIST, reports high accuracy (~99.2%+ for CNN), saves confusion matrix & sample
predictions, and predicts custom hand-drawn digit images.

Usage:
    python digit_recognizer.py                         # train CNN (5 epochs) & evaluate
    python digit_recognizer.py my_digit.png            # predict image (uses saved weights)
    python digit_recognizer.py --model rf              # train & evaluate Random Forest (~96.8%)
    python digit_recognizer.py --epochs 5 --retrain    # force retrain CNN
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

MODEL_WEIGHTS_PATH = "mnist_cnn.pth"


# ---------------------------------------------------------------- 1. DATA
def load_data():
    """Download MNIST (70,000 images, 28x28) and scale pixels to 0-1."""
    print("Loading MNIST dataset...")
    try:
        X, y = fetch_openml(data_id=554, return_X_y=True, as_frame=False, parser="auto")
    except Exception:
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = (X / 255.0).astype(np.float32)
    y = y.astype(np.int64)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ---------------------------------------------------------------- 2. CNN ARCHITECTURE
class DigitCNN(nn.Module):
    """Convolutional Neural Network for robust handwritten digit classification."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 7x7
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


# ---------------------------------------------------------------- 3. TRAINING
def train_cnn(X_train, y_train, X_test, y_test, epochs=5, batch_size=128):
    """Train the CNN on MNIST training split with validation tracking."""
    if os.cpu_count():
        torch.set_num_threads(os.cpu_count())
    print(f"\nTraining CNN model for {epochs} epochs on {len(X_train)} samples...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train_t = torch.from_numpy(X_train.reshape(-1, 1, 28, 28))
    y_train_t = torch.from_numpy(y_train)
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )

    model = DigitCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)

        avg_loss = total_loss / len(X_train)

        # Evaluate on validation split
        model.eval()
        with torch.no_grad():
            X_test_t = torch.from_numpy(X_test.reshape(-1, 1, 28, 28)).to(device)
            val_preds = model(X_test_t).argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_test, val_preds) * 100

        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} - Test Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    print(f"Model checkpoint saved to {MODEL_WEIGHTS_PATH}")
    return model


def train_rf(X_train, y_train):
    """Train Random Forest classifier baseline."""
    print("\nTraining Random Forest baseline (100 estimators)...")
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    return model


# ------------------------------------------------------------ 4. EVALUATION
def evaluate_cnn(model, X_test, y_test):
    """Evaluate CNN accuracy and return predictions."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test.reshape(-1, 1, 28, 28)).to(device)
        preds = model(X_test_t).argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_test, preds) * 100
    print(f"\nFinal CNN Test accuracy: {acc:.2f}%")
    return preds


def evaluate_rf(model, X_test, y_test):
    """Evaluate Random Forest accuracy and return predictions."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    print(f"\nFinal Random Forest Test accuracy: {acc:.2f}%")
    return preds


# ------------------------------------------------------------- 5. PLOTS
def save_confusion_matrix(y_test, preds, path="confusion_matrix.png", title="Confusion Matrix"):
    """Show which digits get mixed up and save the plot."""
    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.title(title)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def save_sample_predictions(X_test, y_test, preds, path="sample_predictions.png"):
    """Save a grid of 10 test images with true vs predicted labels."""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for ax, img, true, pred in zip(axes.ravel(), X_test, y_test, preds):
        ax.imshow(img.reshape(28, 28), cmap="gray")
        color = "green" if true == pred else "red"
        ax.set_title(f"True {true} / Pred {pred}", color=color)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ------------------------------------------------- 6. PREDICT YOUR OWN IMAGE
def predict_image_cnn(model, path):
    """Predict digit and confidence from user image using CNN."""
    device = next(model.parameters()).device
    img = Image.open(path).convert("L")
    img = ImageOps.invert(img)  # Dark digit on light paper -> white on black
    img = img.resize((28, 28))
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
        confidence = float(probs[pred]) * 100

    print(f"\nPrediction for '{path}': Digit {pred} (Confidence: {confidence:.2f}%)")
    return pred


def predict_image_rf(model, path):
    """Predict digit from user image using Random Forest."""
    img = Image.open(path).convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    arr = (np.array(img).astype(np.float32) / 255.0).reshape(1, -1)
    pred = model.predict(arr)[0]
    print(f"\nPrediction for '{path}': Digit {pred}")
    return pred


# ---------------------------------------------------------------- MAIN
def main():
    parser = argparse.ArgumentParser(description="Handwritten Digit Recognizer (MNIST)")
    parser.add_argument("image", nargs="?", default=None, help="Path to custom image to predict")
    parser.add_argument("--model", choices=["cnn", "rf"], default="cnn", help="Model type: 'cnn' or 'rf'")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs for CNN (default: 5)")
    parser.add_argument("--retrain", action="store_true", help="Force retraining even if saved weights exist")
    args = parser.parse_args()

    # If only predicting an image with CNN and weights exist, load weights without retraining
    if args.image and args.model == "cnn" and os.path.exists(MODEL_WEIGHTS_PATH) and not args.retrain:
        print(f"Loading existing CNN weights from '{MODEL_WEIGHTS_PATH}'...")
        model = DigitCNN()
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, weights_only=True))
        predict_image_cnn(model, args.image)
        return

    # Load MNIST dataset
    X_train, X_test, y_train, y_test = load_data()

    if args.model == "cnn":
        # Check if saved model already exists and retrain not requested
        if os.path.exists(MODEL_WEIGHTS_PATH) and not args.retrain and not args.image:
            print(f"Found saved model at '{MODEL_WEIGHTS_PATH}'. Loading...")
            model = DigitCNN()
            model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, weights_only=True))
        else:
            model = train_cnn(X_train, y_train, X_test, y_test, epochs=args.epochs)

        preds = evaluate_cnn(model, X_test, y_test)
        save_confusion_matrix(y_test, preds, title=f"CNN Confusion Matrix ({accuracy_score(y_test, preds)*100:.2f}%)")
        save_sample_predictions(X_test, y_test, preds)

        if args.image:
            predict_image_cnn(model, args.image)

    else:
        # Random Forest Baseline
        model = train_rf(X_train, y_train)
        preds = evaluate_rf(model, X_test, y_test)
        save_confusion_matrix(y_test, preds, title=f"Random Forest Confusion Matrix ({accuracy_score(y_test, preds)*100:.2f}%)")
        save_sample_predictions(X_test, y_test, preds)

        if args.image:
            predict_image_rf(model, args.image)


if __name__ == "__main__":
    main()
