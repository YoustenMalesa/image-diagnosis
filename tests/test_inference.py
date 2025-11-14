from src.inference import predict_image_bytes, _preprocess, severity_and_stage
from PIL import Image
import io

def test_preprocess():
    """Test image preprocessing."""
    img = Image.new('RGB', (100, 100), color='red')
    tensor = _preprocess(img, image_size=224)
    
    assert tensor.shape == (3, 224, 224)

def test_severity_and_stage():
    """Test severity classification logic."""
    severity, stage = severity_and_stage(0.3)
    assert severity == "Low" and stage == "Early"
    
    severity, stage = severity_and_stage(0.6)
    assert severity == "Medium" and stage == "Progressed"
    
    severity, stage = severity_and_stage(0.9)
    assert severity == "High" and stage == "Advanced"

def test_predict_image_bytes():
    """Test prediction on actual image bytes."""
    img = Image.new('RGB', (224, 224), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    result = predict_image_bytes(img_bytes.getvalue())
    
    assert "condition" in result
    assert "probability" in result
    assert 0.0 <= result["probability"] <= 1.0
    assert "severity" in result
    assert "stage" in result