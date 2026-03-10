import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "model"
UPLOAD_DIR = BASE_DIR / "uploads"

# Model settings - EfficientNet-B0 Coconut Disease Classifier
MODEL_PATH = MODEL_DIR / "best_model.pth"
IMAGE_SIZE = 224
NUM_CLASSES = 5          # visible output classes
NUM_RAW_CLASSES = 6      # actual model head output nodes (MUST be 6)
CONFIDENCE_THRESHOLD = 70.0  # below this → flagged as uncertain

# 6-class → 5-class merge map (old index 5 merges into index 3)
OLD_TO_NEW = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 3}

# Class names (5 final classes)
CLASS_NAMES = [
    "CCI_Caterpillars",        # 0
    "CCI_Leaflets",            # 1
    "Healthy_Leaves",          # 2
    "WCLWD_Drying_Yellowing",  # 3  ← merged (old 3 + old 5)
    "WCLWD_Flaccidity",        # 4
]

# Allowed image types
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff"}

# API settings
API_TITLE = "🥥 Coconut Disease Classifier API"
API_DESCRIPTION = """
## 🥥 Coconut Disease Classifier

A FastAPI-powered REST API for classifying **coconut leaf diseases**
using an **EfficientNet-B0** model trained with the `timm` library.

### 🌿 Detectable Conditions:
| Index | Class | Description |
|-------|-------|-------------|
| 0 | `CCI_Caterpillars` | Caterpillar infestation |
| 1 | `CCI_Leaflets` | Leaflet damage from CCI |
| 2 | `Healthy_Leaves` | No disease detected |
| 3 | `WCLWD_Drying_Yellowing` | WCLWD drying & yellowing (merged) |
| 4 | `WCLWD_Flaccidity` | WCLWD flaccidity stage |

### 🔬 Model Specs:
- **Architecture**: EfficientNet-B0 (timm)
- **Input Size**: 224 × 224 RGB
- **Test Accuracy**: 99.87% (preprocessed) / ~90%+ (raw)
- **Confidence Threshold**: 70% (below → uncertain flag)

### 📡 Endpoints:
- `POST /predict` — Single image prediction
- `POST /predict/batch` — Batch prediction (max 20 images)
- `GET  /health` — API health check
- `GET  /info` — Model details
- `GET  /classes` — List all disease classes
"""
API_VERSION = "1.0.0"

# Create upload dir if not exists
os.makedirs(UPLOAD_DIR, exist_ok=True)