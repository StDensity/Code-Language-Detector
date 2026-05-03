from typing import Literal

from pydantic import BaseModel, Field


ModelName = Literal["tfidf", "cnn"]


class PredictRequest(BaseModel):
    code: str = Field(..., min_length=1)
    model: ModelName = "tfidf"


class Prediction(BaseModel):
    model: str
    language: str
    confidence: float
    inference_time_ms: float


class PredictResponse(Prediction):
    pass


class CompareRequest(BaseModel):
    code: str = Field(..., min_length=1)


class CompareResponse(BaseModel):
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    available_models: list[str]


class ModelsResponse(BaseModel):
    available_models: list[str]
