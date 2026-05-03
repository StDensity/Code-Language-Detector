from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.schemas import (
    CompareRequest,
    CompareResponse,
    HealthResponse,
    ModelsResponse,
    PredictRequest,
    PredictResponse,
)
from app.services import InferenceService


inference_service = InferenceService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    inference_service.load_models()
    yield
    inference_service.clear()


app = FastAPI(
    title="Code Language Detector",
    description="Inference API for TF-IDF and char-level CNN code language models.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "available_models": inference_service.available_models(),
    }


@app.get("/models", response_model=ModelsResponse)
def models():
    return {"available_models": inference_service.available_models()}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not inference_service.has_model(request.model):
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Model '{request.model}' is not available.",
                "available_models": inference_service.available_models(),
            },
        )

    return inference_service.predict(request.model, request.code)


@app.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest):
    predictions = inference_service.compare(request.code)
    if not predictions:
        raise HTTPException(
            status_code=404,
            detail="No trained models are available.",
        )

    return {"predictions": predictions}
