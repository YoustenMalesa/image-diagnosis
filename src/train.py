import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from rich import print

from .config import (
    DATA_DIR,
    MODELS_DIR,
    LOGS_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    NUM_WORKERS,
    DEFAULT_MODEL_PATH,
    LOG_INTERVAL,
)
from .data import get_dataloaders
from .model import build_model, save_checkpoint


def train_one_epoch(model, loader: DataLoader, criterion, optimizer, device, scheduler=None) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    if (batch_idx + 1) % LOG_INTERVAL == 0:
        seen = total
        print(f"  [batch {batch_idx+1}/{len(loader)}] loss={total_loss/seen:.4f} acc={correct/seen:.4f}")

    return {"loss": total_loss / total, "acc": correct / total}


def evaluate(model, loader: DataLoader, criterion, device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return {"loss": total_loss / total, "acc": correct / total}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=DATA_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # Print dataset sizes to show progress early
    print(f"Dataset sizes -> train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)}, test: {len(test_loader.dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_model(num_classes=len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Optional OneCycleLR
    steps_per_epoch = max(1, len(train_loader))
    scheduler = OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=steps_per_epoch, epochs=NUM_EPOCHS)

    best_val_acc = 0.0
    patience_counter = 0
    best_path = DEFAULT_MODEL_PATH

    for epoch in range(NUM_EPOCHS):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train loss: {train_metrics['loss']:.4f} acc: {train_metrics['acc']:.4f} | "
              f"Val loss: {val_metrics['loss']:.4f} acc: {val_metrics['acc']:.4f}")

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            patience_counter = 0
            save_checkpoint(model, class_names, str(best_path))
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    # Final test evaluation
    if best_path.exists():
        from .model import load_checkpoint

        best_model, class_names = load_checkpoint(str(best_path))
        best_model = best_model.to(device)
        test_metrics = evaluate(best_model, test_loader, criterion, device)
        print(f"Test loss: {test_metrics['loss']:.4f} acc: {test_metrics['acc']:.4f}")

        # Detailed report
        y_true = []
        y_pred = []
        best_model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = best_model(images)
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())
        print(classification_report(y_true, y_pred, target_names=class_names))
        print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
