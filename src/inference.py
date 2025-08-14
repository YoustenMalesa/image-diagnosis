from io import BytesIO
from typing import Dict, Tuple

import torch
from PIL import Image
from torchvision import transforms

from .model import load_checkpoint
from .config import DEFAULT_MODEL_PATH, IMAGE_SIZE


def _preprocess(image: Image.Image, image_size: int = IMAGE_SIZE):
    tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tfms(image.convert("RGB"))


def severity_and_stage(prob: float) -> Tuple[str, str]:
    # Example policy mapping probability to severity/stage thresholds
    # Adjust realistically for clinical context
    if prob < 0.5:
        return "Low", "Early"
    elif prob < 0.75:
        return "Medium", "Progressed"
    else:
        return "High", "Advanced"


def predict_image_bytes(image_bytes: bytes, model_path: str = str(DEFAULT_MODEL_PATH)) -> Dict:
    model, class_names = load_checkpoint(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    img = Image.open(BytesIO(image_bytes))
    tensor = _preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        top_prob, top_idx = probs.max(dim=1)

    prob = float(top_prob.item())
    cls_idx = int(top_idx.item())
    condition = class_names[cls_idx]
    severity, stage = severity_and_stage(prob)
    return {
        "condition": condition,
        "probability": prob,
        "severity": severity,
        "stage": stage,
        "class_index": cls_idx,
        "class_names": class_names,
    }
