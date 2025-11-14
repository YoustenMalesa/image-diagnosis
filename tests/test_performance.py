import json
import pytest
import torch
from pathlib import Path

def test_model_performance_baseline():
    """Verify model meets minimum performance thresholds."""
    # After training, load model and evaluate
    from src.model import load_checkpoint
    from src.data import get_dataloaders
    from src.train import evaluate
    
    model_path = "models/skin_cnn_resnet18.pt"
    if not Path(model_path).exists():
        pytest.skip("Model not trained yet")
    
    model, _ = load_checkpoint(model_path)
    _, _, test_loader, _ = get_dataloaders()
    
    device = torch.device("cpu")
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    metrics = evaluate(model, test_loader, criterion, device)
    
    # Define minimum acceptable performance
    assert metrics["acc"] >= 0.60, f"Model accuracy {metrics['acc']:.2f} below 60% threshold"