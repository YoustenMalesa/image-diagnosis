from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .config import IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, DATA_DIR, SEED


def build_transforms(image_size: int = IMAGE_SIZE):
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tfms, val_tfms


def get_dataloaders(
    data_dir: Path = DATA_DIR,
    image_size: int = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    val_split: float = 0.15,
    test_split: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    train_tfms, val_tfms = build_transforms(image_size)

    # Create a dataset once to compute class names and file ordering
    base_dataset = datasets.ImageFolder(root=str(data_dir))
    class_names = base_dataset.classes

    # splits
    total_len = len(base_dataset)
    test_len = int(total_len * test_split)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len - test_len

    generator = torch.Generator().manual_seed(SEED)
    # indices into the shared ordering
    train_subset, val_subset, test_subset = random_split(base_dataset, [train_len, val_len, test_len], generator)

    # Now create three datasets with different transforms but same ordering
    train_dataset = datasets.ImageFolder(root=str(data_dir), transform=train_tfms)
    val_dataset = datasets.ImageFolder(root=str(data_dir), transform=val_tfms)
    test_dataset = datasets.ImageFolder(root=str(data_dir), transform=val_tfms)

    # Wrap them with Subset using the indices computed from base_dataset
    from torch.utils.data import Subset

    train_ds = Subset(train_dataset, train_subset.indices)
    val_ds = Subset(val_dataset, val_subset.indices)
    test_ds = Subset(test_dataset, test_subset.indices)

    # Pin memory only when CUDA is available; persistent workers only when workers > 0
    pin = torch.cuda.is_available()
    persistent = num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin, persistent_workers=persistent)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin, persistent_workers=persistent)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin, persistent_workers=persistent)

    return train_loader, val_loader, test_loader, class_names
