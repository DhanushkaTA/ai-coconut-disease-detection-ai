import os
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from config import IMAGE_SIZE, SUPPORTED_FORMATS, LOG_FILE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_image(image_path):
    """
    Load and validate image from path
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        PIL.Image or None: Loaded image or None if invalid
    """
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
            
        file_ext = Path(image_path).suffix.lower()
        if file_ext not in SUPPORTED_FORMATS:
            logger.error(f"Unsupported image format: {file_ext}")
            return None
            
        img = Image.open(image_path).convert('RGB')
        logger.info(f"Successfully loaded image: {image_path}")
        return img
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None


def preprocess_image(image):
    """
    Preprocess image for model prediction
    
    Args:
        image (PIL.Image): PIL Image object
        
    Returns:
        np.ndarray: Preprocessed image array ready for model
    """
    try:
        # Resize image
        img_resized = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img_resized, dtype=np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None


def get_image_info(image_path):
    """
    Get image file information
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        dict: Image information (size, file size, etc.)
    """
    try:
        img = Image.open(image_path)
        file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
        
        return {
            "path": image_path,
            "size": img.size,
            "format": img.format,
            "file_size_mb": round(file_size, 2)
        }
    except Exception as e:
        logger.error(f"Error getting image info: {str(e)}")
        return None


def format_confidence(confidence):
    """
    Format confidence score as percentage
    
    Args:
        confidence (float): Confidence score (0-1)
        
    Returns:
        str: Formatted confidence percentage
    """
    return f"{confidence * 100:.2f}%"
