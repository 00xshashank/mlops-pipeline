# datasets/dataloaders.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from datasets.augmentations import custom_augment

IMG_SIZE = 224
BATCH_SIZE = 32

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0))
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def create_dataloaders(base_dir: str):
    base_dir = Path(base_dir)

    train_ds = datasets.ImageFolder(base_dir / "train", transform=train_transform)
    val_ds   = datasets.ImageFolder(base_dir / "val", transform=eval_transform)
    test_ds  = datasets.ImageFolder(base_dir / "test", transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, len(train_ds.classes)
