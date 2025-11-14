from fastapi.testclient import TestClient
from app.main import app
from PIL import Image
import io

client = TestClient(app)

def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_valid_image():
    """Test prediction with valid image."""
    img = Image.new('RGB', (224, 224), color='green')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    response = client.post("/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
    assert response.status_code == 200
    
    data = response.json()
    assert "condition" in data
    assert "probability" in data

def test_predict_invalid_file():
    """Test prediction with non-image file."""
    response = client.post("/predict", files={"file": ("test.txt", b"not an image", "text/plain")})
    assert response.status_code == 400