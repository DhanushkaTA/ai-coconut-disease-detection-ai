from pydantic import BaseModel
from typing import List, Optional, Dict


class PredictionItem(BaseModel):
    rank: int
    disease: str
    confidence: float
    confidence_percent: str


class PredictionResponse(BaseModel):
    success: bool
    filename: str
    predicted_disease: str
    class_index: int
    confidence: float
    confidence_percent: str
    uncertain: bool                        # True if confidence < 70%
    all_probabilities: Dict[str, float]    # All 5 class probabilities (%)
    top_5_predictions: List[PredictionItem]
    processing_time_ms: float


class BatchPredictionResponse(BaseModel):
    success: bool
    total_images: int
    results: List[PredictionResponse]
    failed: List[Dict[str, str]]
    total_processing_time_ms: float


class ModelInfoResponse(BaseModel):
    success: bool
    model_path: str
    architecture: str
    num_classes: int
    num_raw_classes: int
    image_size: int
    device: str
    confidence_threshold: float
    total_parameters: int
    trainable_parameters: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    api_version: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None