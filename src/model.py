from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int) -> Tuple[nn.Module, int]:
    """Return a pretrained ResNet18 adapted for num_classes and the input size."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )
    return model, 224


def save_checkpoint(model: nn.Module, class_names: list, path: str):
    payload = {
        "state_dict": model.state_dict(),
        "class_names": class_names,
    }
    torch.save(payload, path)


def load_checkpoint(path: str) -> Tuple[nn.Module, list]:
    payload = torch.load(path, map_location="cpu")
    class_names = payload["class_names"]
    model, _ = build_model(num_classes=len(class_names))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, class_names
