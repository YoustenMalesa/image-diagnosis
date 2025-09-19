# Skin Condition Diagnosis (CNN + FastAPI)

End-to-end training and inference for skin condition classification using a ResNet18 CNN, wrapped in a FastAPI service. Dataset is read from `./data/<class_name>/*.jpg` where folder names are the labels. The API returns condition, probability, severity, and stage.

## Project layout

- `data/` — Your dataset (not copied into the Docker image)
- `models/` — Saved model checkpoints (bind-mounted into the container)
- `logs/` — Training logs
- `src/` — Training and inference code
- `app/` — FastAPI app entrypoint
- `Dockerfile`, `docker-compose.yml`, `.dockerignore`

## Setup (local)

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; . .\.venv\Scripts\Activate.ps1; pip install --upgrade pip; pip install -r requirements.txt  
```

2. Train locally:

```powershell
# Uses data/ as source, saves model to models/skin_cnn_resnet18.pt
python -m src.train
```

3. Run the API locally:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000/docs to try `/predict`.

## Docker

- The dataset is excluded from build context via `.dockerignore`.
- Build-time training uses a bind mount (BuildKit) so the dataset is not copied into the image.

Build and run with Docker Compose (requires BuildKit):

```powershell
$env:DOCKER_BUILDKIT=1; docker compose build
# Run the API with dataset and models mounted from host
docker compose up api
```

One-off training container:

```powershell
docker compose run --rm train
```

## API

## Build and run: docker compose up -d --build 

POST /predict (multipart/form-data)
- file: image

Response:
```json
{
  "condition": "Melanoma",
  "probability": 0.93,
  "severity": "High",
  "stage": "Advanced",
  "class_index": 3,
  "class_names": ["Actinic keratosis", "Benign keratosis", ...]
}
```

## Notes
- Severity/Stage is derived from top-class probability with simple thresholds: <0.5 Low/Early, <0.75 Medium/Progressed, else High/Advanced. Adjust as needed.
- For GPU training, switch to a CUDA base image and install appropriate PyTorch builds.
- If classes are imbalanced, consider WeightedRandomSampler or class weights.

## commands
## Commands
Build & Train: docker build -t image-diagnosis:latest .
Run: docker run -d --name image-diagnosis-model --network mobiclinic-net -p 8001:8000 yousten/image-diagnosis-model:latest
