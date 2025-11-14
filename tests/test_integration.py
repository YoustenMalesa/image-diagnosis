from src.train import train_one_epoch, evaluate
from src.data import get_dataloaders
from src.model import build_model
import torch

def test_train_evaluate_pipeline():
    """Test complete training and evaluation loop."""
    # Use small batch for testing
    train_loader, val_loader, _, class_names = get_dataloaders(batch_size=8)
    
    device = torch.device("cpu")
    model, _ = build_model(len(class_names))
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Train one epoch
    metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
    assert "loss" in metrics and "acc" in metrics
    assert metrics["loss"] > 0
    assert 0.0 <= metrics["acc"] <= 1.0
    
    # Evaluate
    val_metrics = evaluate(model, val_loader, criterion, device)
    assert "loss" in val_metrics and "acc" in val_metrics