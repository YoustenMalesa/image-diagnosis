from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from src.config import DEFAULT_MODEL_PATH
from src.inference import predict_image_bytes

app = FastAPI(title="Skin Condition Diagnosis API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    image_bytes = await file.read()

    if not DEFAULT_MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not available. Please train the model first.")

    try:
        out = predict_image_bytes(image_bytes, model_path=str(DEFAULT_MODEL_PATH))
        return JSONResponse(content=out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
