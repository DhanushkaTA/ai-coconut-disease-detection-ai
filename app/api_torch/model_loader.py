import torch
import logging
from pathlib import Path

try:
    import timm
except ImportError:
    raise ImportError("❌ 'timm' not installed. Run: pip install timm")

from config import MODEL_PATH, NUM_RAW_CLASSES

logger = logging.getLogger(__name__)


class ModelManager:
    """Handles EfficientNet-B0 model loading via timm"""

    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.architecture = "EfficientNet-B0 (timm)"
        self._load_model()

    def _load_model(self):
        """Load EfficientNet-B0 weights from .pth file"""
        model_path = Path(MODEL_PATH)

        if not model_path.exists():
            raise FileNotFoundError(f"❌ Model not found at: {model_path}")

        logger.info(f"Loading model from : {model_path}")
        logger.info(f"Device             : {self.device}")

        # ⚠️ MUST use num_classes=6 — the saved head has 6 output nodes
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=NUM_RAW_CLASSES  # 6, not 5!
        )

        state_dict = torch.load(model_path, map_location=self.device)

        # Handle wrapped checkpoints
        if isinstance(state_dict, dict):
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            # else: assume it's already a plain state_dict

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.info("✅ EfficientNet-B0 loaded successfully!")

    def get_model(self):
        return self.model

    def is_loaded(self) -> bool:
        return self.model is not None

    def get_device(self) -> str:
        return str(self.device)

    def get_total_params(self) -> int:
        if self.model:
            return sum(p.numel() for p in self.model.parameters())
        return 0

    def get_trainable_params(self) -> int:
        if self.model:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return 0