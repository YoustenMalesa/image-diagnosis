import torch
from src.model import build_model, save_checkpoint, load_checkpoint
import tempfile

def test_build_model():
    """Test model architecture and output shape."""
    num_classes = 11
    model, input_size = build_model(num_classes)
    assert input_size == 224
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    assert output.shape == (2, num_classes)

def test_save_and_load_checkpoint():
    """Test checkpoint persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model, _ = build_model(5)
        class_names = ["Class1", "Class2", "Class3", "Class4", "Class5"]
        path = f"{tmpdir}/model.pt"
        
        save_checkpoint(model, class_names, path)
        loaded_model, loaded_names = load_checkpoint(path)
        
        assert loaded_names == class_names
        assert isinstance(loaded_model, torch.nn.Module)