import torch
import torch.nn.functional as F
import cv2
import numpy as np
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

from model_loader import ModelManager
from config import (
    IMAGE_SIZE,
    CLASS_NAMES,
    OLD_TO_NEW,
    NUM_CLASSES,
    CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


class Predictor:
    """Handles all preprocessing and prediction logic"""

    def __init__(self):
        self.model_manager = ModelManager()
        self.classes = CLASS_NAMES
        self.device = self.model_manager.device

    # ─── Preprocessing (mirrors training pipeline) ────────────────────────────
    def preprocess(self, img_path: str) -> torch.Tensor:
        """
        Exact preprocessing pipeline used during training:
          1. Read with OpenCV
          2. Auto-crop to bounding box (remove white borders)
          3. Gaussian blur
          4. Pad to square → resize to 224×224
          5. Normalize to [0,1] → tensor
        """
        img = cv2.imread(img_path)

        # Fallback: try PIL if OpenCV fails (WEBP, HEIC, etc.)
        if img is None:
            try:
                from PIL import Image
                pil_img = Image.open(img_path).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception:
                raise ValueError(f"Cannot read image: {img_path}")

        # Step 1 — Auto-crop to content bounding box
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            x, y, w, h = cv2.boundingRect(np.concatenate(contours))
            img = img[y: y + h, x: x + w]

        # Step 2 — Gaussian blur
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # Step 3 — Pad to square, then resize
        h, w = img.shape[:2]
        size = max(h, w)
        padded = np.full((size, size, 3), 255, dtype=np.uint8)
        padded[(size - h) // 2: (size - h) // 2 + h,
               (size - w) // 2: (size - w) // 2 + w] = img
        img = cv2.resize(padded, (IMAGE_SIZE, IMAGE_SIZE))

        # Step 4 — Normalize to [0, 1] and convert to tensor [1, C, H, W]
        tensor = torch.from_numpy(img.astype("float32") / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWC → CHW → NCHW
        return tensor.to(self.device)

    # ─── 6-class → 5-class merge ──────────────────────────────────────────────
    def merge_classes(self, probs_6: torch.Tensor) -> torch.Tensor:
        """Merge 6 raw output probabilities into 5 final classes"""
        probs_5 = torch.zeros(1, NUM_CLASSES, device=self.device)
        for old_idx, new_idx in OLD_TO_NEW.items():
            probs_5[:, new_idx] += probs_6[:, old_idx]
        return probs_5

    # ─── Single prediction ────────────────────────────────────────────────────
    def predict_single(self, image_path: str) -> Dict[str, Any]:
        start = time.time()

        try:
            model = self.model_manager.get_model()
            tensor = self.preprocess(image_path)

            with torch.no_grad():
                raw_output = model(tensor)
                probs_6 = F.softmax(raw_output, dim=1)
                probs_5 = self.merge_classes(probs_6)

            probs_np = probs_5.cpu().numpy()[0]

            pred_idx = int(np.argmax(probs_np))
            confidence = float(probs_np[pred_idx])
            uncertain = (confidence * 100) < CONFIDENCE_THRESHOLD

            # All class probabilities
            all_probs = {
                CLASS_NAMES[i]: round(float(probs_np[i]) * 100, 2)
                for i in range(NUM_CLASSES)
            }

            # Top-5 predictions (sorted)
            sorted_idx = np.argsort(probs_np)[::-1]
            top_5 = [
                {
                    "rank": rank + 1,
                    "disease": CLASS_NAMES[i],
                    "confidence": round(float(probs_np[i]), 4),
                    "confidence_percent": f"{float(probs_np[i]) * 100:.2f}%",
                }
                for rank, i in enumerate(sorted_idx)
            ]

            elapsed_ms = round((time.time() - start) * 1000, 2)

            return {
                "success": True,
                "filename": Path(image_path).name,
                "predicted_disease": CLASS_NAMES[pred_idx],
                "class_index": pred_idx,
                "confidence": round(confidence, 4),
                "confidence_percent": f"{confidence * 100:.2f}%",
                "uncertain": uncertain,
                "all_probabilities": all_probs,
                "top_5_predictions": top_5,
                "processing_time_ms": elapsed_ms,
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "success": False,
                "filename": Path(image_path).name,
                "error": str(e),
            }

    # ─── Batch prediction ─────────────────────────────────────────────────────
    def predict_batch(self, image_paths: List[str]) -> Dict[str, Any]:
        start = time.time()
        results, failed = [], []

        for path in image_paths:
            result = self.predict_single(path)
            if result.get("success"):
                results.append(result)
            else:
                failed.append({
                    "file": Path(path).name,
                    "error": result.get("error", "Unknown error")
                })

        return {
            "success": True,
            "total_images": len(image_paths),
            "results": results,
            "failed": failed,
            "total_processing_time_ms": round((time.time() - start) * 1000, 2),
        }