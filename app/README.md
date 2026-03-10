# 🥥 Coconut Disease Classifier API

A production-ready **FastAPI** REST application for detecting coconut leaf diseases
using **EfficientNet-B0** (timm) with **Swagger UI** built-in.

---

## ⚡ Quick Setup (5 Minutes)

### 1️⃣ Clone / Navigate to Project
```bash
cd /home/kasr/Storage/tharindu_works/AI/app
```

### 2️⃣ Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install fastapi "uvicorn[standard]" torch torchvision timm \
            opencv-python pillow python-multipart
```

### 4️⃣ Verify Model File Exists
```bash
ls model/best_model.pth
```
> ✅ Expected: `model/best_model.pth` (~665 MB)

### 5️⃣ Start the API Server
```bash
cd api_torch
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 6️⃣ Open Swagger UI
```
http://localhost:8000/docs
```

---

## 📁 Project Structure

```
app/
├── model/
│   └── best_model.pth          # EfficientNet-B0 weights (~665 MB)
│
├── api_torch/
│   ├── main.py                 # FastAPI app + all routes
│   ├── config.py               # Paths, class names, thresholds
│   ├── model_loader.py         # timm model loading (num_classes=6)
│   ├── predictor.py            # Preprocessing + 6→5 class merge logic
│   └── schemas.py              # Pydantic request/response models
│
├── uploads/                    # Temp folder for uploaded images (auto-created)
└── README.md
```

---

## 🌐 API Endpoints

| Method | Endpoint         | Description                        |
|--------|------------------|------------------------------------|
| GET    | `/docs`          | **Swagger UI** (interactive)       |
| GET    | `/redoc`         | ReDoc documentation                |
| GET    | `/health`        | API & model health status          |
| GET    | `/info`          | Model architecture details         |
| GET    | `/classes`       | List all 5 disease classes         |
| POST   | `/predict`       | Predict single image               |
| POST   | `/predict/batch` | Predict up to 20 images at once    |

---

## 🌿 Disease Classes

| Index | Class | Description |
|-------|-------|-------------|
| 0 | `CCI_Caterpillars` | Caterpillar infestation |
| 1 | `CCI_Leaflets` | Leaflet damage from CCI |
| 2 | `Healthy_Leaves` | No disease detected |
| 3 | `WCLWD_Drying_Yellowing` | WCLWD drying & yellowing (merged) |
| 4 | `WCLWD_Flaccidity` | WCLWD flaccidity stage |

---

## 🧪 Test the API

### Using Swagger UI
1. Go to `http://localhost:8000/docs`
2. Click **POST /predict → Try it out**
3. Upload a coconut leaf image
4. Click **Execute**

### Using curl
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -F "image=@/path/to/coconut_leaf.jpg"

# Model info
curl http://localhost:8000/info
```

### Example Response
```json
{
  "success": true,
  "filename": "leaf.jpg",
  "predicted_disease": "WCLWD_Flaccidity",
  "class_index": 4,
  "confidence": 0.9437,
  "confidence_percent": "94.37%",
  "uncertain": false,
  "all_probabilities": {
    "CCI_Caterpillars": 0.21,
    "CCI_Leaflets": 0.08,
    "Healthy_Leaves": 1.04,
    "WCLWD_Drying_Yellowing": 4.30,
    "WCLWD_Flaccidity": 94.37
  },
  "top_5_predictions": [...],
  "processing_time_ms": 34.5
}
```

---

## 🔍 Confidence Guide

| Confidence | Status | Action |
|------------|--------|--------|
| ≥ 90% | ✅ High | Reliable prediction |
| 70–89% | ✓ Good | Mostly reliable |
| < 70% | ⚠️ Uncertain | Retake photo in better lighting |

> When `"uncertain": true` is returned, prompt user to retake the photo.

---

## ⚙️ Configuration (`config.py`)

```python
MODEL_PATH           = "model/best_model.pth"
IMAGE_SIZE           = 224          # Input size (px)
NUM_CLASSES          = 5            # Final output classes
NUM_RAW_CLASSES      = 6            # Model head size (DO NOT CHANGE)
CONFIDENCE_THRESHOLD = 70.0         # Uncertain flag threshold (%)
```

---

## 🐛 Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'timm'`
```bash
pip install timm
```

### ❌ `RuntimeError: size mismatch`
The model head has **6 outputs**, not 5.
Ensure `NUM_RAW_CLASSES = 6` in `config.py` — do **not** change this.

### ❌ `FileNotFoundError: Model not found`
```bash
ls model/best_model.pth   # Verify file exists
```

### ❌ `cv2 cannot read the image`
Convert to JPG first:
```bash
python -c "from PIL import Image; Image.open('file.heic').convert('RGB').save('file.jpg')"
```

### ❌ Slow predictions (2–3 seconds)
Running on CPU. GPU inference is ~30ms vs ~2s on CPU.
Check GPU availability:
```python
import torch; print(torch.cuda.is_available())
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `uvicorn[standard]` | ASGI server |
| `timm` | EfficientNet-B0 model |
| `torch` + `torchvision` | Deep learning |
| `opencv-python` | Image preprocessing |
| `pillow` | Image fallback reader |
| `python-multipart` | File upload support |

---

## 🚀 Production Deployment

```bash
# Single worker (prevents 665MB model loading multiple times)
pip install gunicorn gevent
gunicorn -w 1 --worker-class gevent -b 0.0.0.0:8000 main:app
```

> ⚠️ Use `-w 1` only — multiple workers each load the 665MB model separately.

---

**Ready? Start with:**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000