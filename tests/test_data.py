# tests/test_data.py
from src.data import get_dataloaders, build_transforms
from pathlib import Path

def test_data_loading():
    """Test dataloader creation and batch integrity."""
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    
    assert len(class_names) > 0
    assert len(train_loader) > 0
    
    # Test batch structure
    images, labels = next(iter(train_loader))
    assert images.shape[0] <= 32  # batch size
    assert images.shape[1] == 3   # channels
    assert images.shape[2] == 224 # height
    assert images.shape[3] == 224 # width
    assert labels.shape[0] == images.shape[0]

def test_data_split():
    """Verify train/val/test split consistency."""
    train_loader, val_loader, test_loader, _ = get_dataloaders()
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)
    
    total = train_size + val_size + test_size
    assert abs(train_size / total - 0.70) < 0.02  # 70% training
    assert abs(val_size / total - 0.15) < 0.02    # 15% validation
    assert abs(test_size / total - 0.15) < 0.02   # 15% testing

def test_transforms():
    """Test image transformation pipeline."""
    train_tfms, val_tfms = build_transforms(224)
    assert train_tfms is not None
    assert val_tfms is not None
    
    from PIL import Image
    import numpy as np
    dummy_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    
    tensor = train_tfms(dummy_img)
    assert tensor.shape == (3, 224, 224)