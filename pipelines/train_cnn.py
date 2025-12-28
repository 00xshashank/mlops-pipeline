from typing import Dict, Tuple, List
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np

from zenml import step, pipeline

import wandb
# wandb.init()

import os
os.environ["PYTHONUTF8"] = "1"
os.environ["ZENML_DISABLE_ARTIFACT_LOGS"] = "true"
os.environ["RICH_DISABLE"] = "1"


from cnn import SkinDiseaseCNN

@step
def initialize_model(num_classes: int) -> nn.Module:
    return SkinDiseaseCNN(num_classes=num_classes)


@step(experiment_tracker="wandb_tracker")
def train_one_epoch(
    model: nn.Module,
    data_path: str,
    lr: float = 1e-4,
    batch_size: int = 32,
) -> tuple[nn.Module, float]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    wandb.log({"train_loss": avg_loss})
    return model.cpu(), avg_loss


@step(experiment_tracker="wandb_tracker")
def evaluate(
    model: nn.Module,
    data_path: str,
    batch_size: int = 32,
) -> Dict[str, float]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size)

    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    wandb.log({"accuracy":accuracy})
    return {"accuracy": accuracy}

@step
def save_model_locally(model: nn.Module):
    import torch
    torch.save(model.state_dict(), "skin_disease_cnn_2.pt")

@pipeline()
def skin_disease_training_pipeline_final(
    data_dir: str,
    epochs: int = 20,
):
    paths = {
        "train": f"{data_dir}/train",
        "val": f"{data_dir}/val",
    }

    model = initialize_model(num_classes=10)

    for _ in range(epochs):
        model, _ = train_one_epoch(
            model=model,
            data_path=paths["train"],
        )

        _ = evaluate(
            model=model,
            data_path=paths["val"],
        )

    save_model_locally(model=model)


if __name__ == "__main__":
    skin_disease_training_pipeline_final(
        data_dir="data",
        epochs=30,
    )
