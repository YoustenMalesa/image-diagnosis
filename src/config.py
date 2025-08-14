import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))
LOGS_DIR = Path(os.getenv("LOGS_DIR", PROJECT_ROOT / "logs"))

# Training defaults
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 224))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 10))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 3e-4))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 1e-4))
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 3))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 0))
SEED = int(os.getenv("SEED", 42))
LOG_INTERVAL = int(os.getenv("LOG_INTERVAL", 50))

# Model file
MODEL_NAME = os.getenv("MODEL_NAME", "skin_cnn_resnet18.pt")
DEFAULT_MODEL_PATH = MODELS_DIR / MODEL_NAME
