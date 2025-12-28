import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple

from zenml import step, pipeline

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import joblib
from tqdm import tqdm

class SkinDiseaseCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            # Block 3 (ENDS AT 128)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.4),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x, return_features=False):
        x = self.features(x)
        x = self.gap(x).flatten(1)

        features = F.relu(self.fc1(x))
        features = self.dropout(features)
        out = self.fc_out(features)

        if return_features:
            return out, features
        return out

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
def load_frozen_cnn(num_classes: int, weights_path: str) -> nn.Module:
    model = SkinDiseaseCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)

    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    return model


@step
def extract_features(
    model: nn.Module,
    data_dir: str,
    split: str,
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
def train_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    model_dir: str,
    n_estimators: int = 300,
    max_depth: int | None = None,
):
    os.makedirs(model_dir, exist_ok=True)

    rf = Pipeline([
        ("scaler", StandardScaler()),  # keeps behavior consistent across models
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        )),
    ])

    rf.fit(X, y)

    model_path = os.path.join(model_dir, "random_forest_model.joblib")
    joblib.dump(rf, model_path)

    print(f"Random Forest model saved to: {model_path}")
    return rf


@step
def evaluate_model(
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


# ============================================================
# Pipeline
# ============================================================
@pipeline
def cnn_random_forest_pipeline(
    data_dir: str,
    num_classes: int,
    cnn_weights: str,
    model_dir: str,
):
    cnn = load_frozen_cnn(num_classes, cnn_weights)

    X_train, y_train = extract_features(cnn, data_dir, "train")
    X_val, y_val     = extract_features(cnn, data_dir, "val")

    rf = train_random_forest(X_train, y_train, model_dir)
    metrics = evaluate_model(rf, X_val, y_val)

    return metrics

if __name__ == "__main__":
    cnn_random_forest_pipeline(
        data_dir="data",
        num_classes=10,
        cnn_weights="skin_disease_cnn_2.pt",
        model_dir="artifacts/models",
    )
