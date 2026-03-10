"""
Coconut Disease Classifier - FastAPI Application
Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
Swagger UI : http://localhost:8000/docs
ReDoc      : http://localhost:8000/redoc
"""

import sys
import shutil
import logging
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent))

from predictor import Predictor
from schemas import (
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfoResponse,
    HealthResponse,
)
from config import (
    API_TITLE, API_DESCRIPTION, API_VERSION,
    ALLOWED_EXTENSIONS, UPLOAD_DIR,
    MODEL_PATH, NUM_CLASSES, NUM_RAW_CLASSES,
    IMAGE_SIZE, CONFIDENCE_THRESHOLD,
)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Startup ──────────────────────────────────────────────────────────────────
predictor: Predictor = None


@app.on_event("startup")
async def startup_event():
    global predictor
    logger.info("🚀 Starting Coconut Disease Classifier API...")
    try:
        predictor = Predictor()
        logger.info("✅ EfficientNet-B0 model loaded and ready!")
    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")


# ─── Helpers ──────────────────────────────────────────────────────────────────
def validate_image(file: UploadFile) -> None:
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type '{ext}' not allowed. Allowed: {ALLOWED_EXTENSIONS}",
        )


def save_upload(file: UploadFile) -> Path:
    ext = file.filename.rsplit(".", 1)[-1].lower()
    path = Path(UPLOAD_DIR) / f"{uuid.uuid4().hex}.{ext}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return path


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "🥥 Coconut Disease Classifier API",
        "swagger_ui": "http://localhost:8000/docs",
        "redoc":      "http://localhost:8000/redoc",
        "health":     "http://localhost:8000/health",
    }


@app.get("/health", response_model=HealthResponse,
         tags=["System"], summary="Health Check")
async def health_check():
    loaded = predictor and predictor.model_manager.is_loaded()
    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=bool(loaded),
        device=predictor.model_manager.get_device() if predictor else "unknown",
        api_version=API_VERSION,
    )


@app.get("/info", response_model=ModelInfoResponse,
         tags=["Model"], summary="Model Information")
async def model_info():
    if not predictor or not predictor.model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    mm = predictor.model_manager
    return ModelInfoResponse(
        success=True,
        model_path=str(MODEL_PATH),
        architecture=mm.architecture,
        num_classes=NUM_CLASSES,
        num_raw_classes=NUM_RAW_CLASSES,
        image_size=IMAGE_SIZE,
        device=mm.get_device(),
        confidence_threshold=CONFIDENCE_THRESHOLD,
        total_parameters=mm.get_total_params(),
        trainable_parameters=mm.get_trainable_params(),
    )


@app.get("/classes", tags=["Model"], summary="List Disease Classes")
async def list_classes():
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "success": True,
        "total_classes": len(predictor.classes),
        "confidence_threshold_pct": CONFIDENCE_THRESHOLD,
        "classes": predictor.classes,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict Single Image",
    description="""
Upload a single coconut leaf image to detect disease.

**Returns:**
- `predicted_disease` — disease class name
- `confidence_percent` — model confidence
- `uncertain` — `true` if confidence < 70% (retake photo recommended)
- `all_probabilities` — probability for all 5 classes
- `top_5_predictions` — ranked list of all predictions
    """,
)
async def predict_single(
    image: UploadFile = File(..., description="Coconut leaf image (JPG/PNG)")
):
    if not predictor or not predictor.model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    validate_image(image)
    saved = save_upload(image)

    try:
        result = predictor.predict_single(str(saved))
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error"))
        return PredictionResponse(**result)
    finally:
        if saved.exists():
            saved.unlink()


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Predict Multiple Images",
    description="Upload up to 20 coconut leaf images for batch prediction.",
)
async def predict_batch(
    images: List[UploadFile] = File(..., description="Multiple coconut leaf images")
):
    if not predictor or not predictor.model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(images) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 images per batch")

    saved_paths = []
    for img in images:
        validate_image(img)
        saved_paths.append(save_upload(img))

    try:
        result = predictor.predict_batch([str(p) for p in saved_paths])
        return BatchPredictionResponse(**result)
    finally:
        for p in saved_paths:
            if p.exists():
                p.unlink()