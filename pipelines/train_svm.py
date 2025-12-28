# train_cnn_svm_zenml.py
import torch
import numpy as np
from pathlib import Path
from typing import Tuple

from zenml import step, pipeline

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from cnn import SkinDiseaseCNN
from tqdm import tqdm

IMG_SIZE = 224
BATCH_SIZE = 32

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

@step
def load_frozen_cnn(num_classes: int, weights_path: str) -> torch.nn.Module:
    model = SkinDiseaseCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    return model


@step
def extract_features(
    model: torch.nn.Module,
    data_dir: str,
    split: str
) -> Tuple[np.ndarray, np.ndarray]:

    base_dir = Path(data_dir)
    transform = train_transform if split == "train" else eval_transform

    dataset = datasets.ImageFolder(base_dir / split, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=(split == "train"),
        num_workers=4,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    features, labels = [], []

    with torch.no_grad():
        for images, y in tqdm(loader, desc=f"Extracting {split} features"):
            images = images.to(device)
            _, feats = model(images, return_features=True)
            features.append(feats.cpu().numpy())
            labels.append(y.numpy())

    return np.concatenate(features), np.concatenate(labels)


@step
def train_svm(X: np.ndarray, y: np.ndarray):
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
        )),
    ])

    svm.fit(X, y)
    return svm


@step
def evaluate_svm(
    model,
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    preds = model.predict(X)

    metrics = {
        "accuracy": accuracy_score(y, preds),
        "report": classification_report(y, preds, output_dict=True),
    }

    print("Accuracy:", metrics["accuracy"])
    return metrics

import os
import joblib
from zenml import step
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@step
def train_svm(
    X: np.ndarray,
    y: np.ndarray,
    model_dir: str,
):
    os.makedirs(model_dir, exist_ok=True)

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
        )),
    ])

    svm.fit(X, y)

    model_path = os.path.join(model_dir, "svm_model.joblib")
    joblib.dump(svm, model_path)

    print(f"SVM model saved to: {model_path}")

    return svm

@pipeline
def cnn_svm_pipeline(
    data_dir: str,
    num_classes: int,
    cnn_weights: str,
    model_dir: str,
):
    cnn = load_frozen_cnn(num_classes, cnn_weights)

    X_train, y_train = extract_features(cnn, data_dir, split="train")
    X_val, y_val     = extract_features(cnn, data_dir, split="val")

    svm = train_svm(X_train, y_train, model_dir)
    metrics = evaluate_svm(svm, X_val, y_val)

    return metrics

if __name__ == "__main__":
    cnn_svm_pipeline(
        data_dir="data",
        num_classes=10,
        cnn_weights="skin_disease_cnn_2.pt",
        model_dir=".",
    )
