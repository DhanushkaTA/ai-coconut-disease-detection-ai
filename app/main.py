#!/usr/bin/env python3
"""
Plant Disease Classifier - Main CLI Application
"""

import sys
import logging
from pathlib import Path
from predictor import Predictor
from config import MODEL_PATH, LABEL_ENCODER_PATH

logger = logging.getLogger(__name__)


def print_header(title):
    """Print formatted header"""
    print("=" * 70)
    print(f"{title:^70}")
    print("=" * 70)


def print_prediction_result(result):
    """Pretty print prediction result"""
    if not result.get("success"):
        print(f"❌ Error: {result.get('error', 'Unknown error')}")
        return
    
    print_header("PREDICTION RESULT")
    print(f"\n📷 Image: {result['image_path']}")
    
    # Image info
    if result.get('image_info'):
        info = result['image_info']
        print(f"\nImage Information:")
        print(f"  Size: {info['size'][0]}x{info['size'][1]} pixels")
        print(f"  File Size: {info['file_size_mb']} MB")
    
    # Main prediction
    print(f"\n{'PREDICTED DISEASE':^70}")
    print(f"Disease: {result['predicted_disease']}")
    print(f"Confidence: {result['confidence_str']}")
    print(f"Processing Time: {result['processing_time']} seconds")
    
    # Top-k predictions
    print(f"\n{'TOP 5 PREDICTIONS':^70}")
    for pred in result['top_predictions']:
        dots = "." * (50 - len(pred['disease']))
        print(f"{pred['rank']}. {pred['disease']}{dots} {pred['confidence_str']}")
    
    print("\n" + "=" * 70 + "\n")


def cmd_predict(image_path):
    """Single image prediction"""
    print()
    predictor = Predictor()
    
    if not predictor.model_manager.is_loaded():
        print("❌ Error: Model not loaded. Check model files exist.")
        return
    
    result = predictor.predict_single_image(image_path)
    print_prediction_result(result)


def cmd_batch(folder_path):
    """Batch prediction on folder"""
    print()
    predictor = Predictor()
    
    if not predictor.model_manager.is_loaded():
        print("❌ Error: Model not loaded. Check model files exist.")
        return
    
    results = predictor.predict_batch(folder_path)
    
    if not results:
        print("❌ No images found in folder")
        return
    
    print_header("BATCH PREDICTION RESULTS")
    successful = 0
    for result in results:
        if result.get('success'):
            successful += 1
            print(f"✅ {result['image_path']}: {result['predicted_disease']} ({result['confidence_str']})")
        else:
            print(f"❌ {result['image_path']}: {result.get('error')}")
    
    print(f"\n{successful}/{len(results)} images processed successfully\n")


def cmd_info():
    """Show model information"""
    print()
    print_header("MODEL INFORMATION")
    
    predictor = Predictor()
    
    # Check files
    print("\n📁 File Status:")
    print(f"  Model: {'✅ Found' if MODEL_PATH.exists() else '❌ Not Found'} ({MODEL_PATH})")
    print(f"  Label Encoder: {'✅ Found' if LABEL_ENCODER_PATH.exists() else '❌ Not Found'} ({LABEL_ENCODER_PATH})")
    
    if not predictor.model_manager.is_loaded():
        print("\n❌ Model not loaded. Check file paths above.")
        return
    
    # Model info
    model = predictor.model_manager.get_model()
    print(f"\n🧠 Model Architecture:")
    print(f"  Input Shape: {model.input_shape}")
    print(f"  Output Shape: {model.output_shape}")
    print(f"  Total Parameters: {model.count_params():,}")
    
    # Classes
    classes = predictor.model_manager.get_classes()
    print(f"\n🌱 Disease Classes ({len(classes)} total):")
    for idx, cls in enumerate(classes[:10], 1):
        print(f"  {idx}. {cls}")
    if len(classes) > 10:
        print(f"  ... and {len(classes) - 10} more")
    
    print("\n" + "=" * 70 + "\n")


def cmd_interactive():
    """Interactive prediction mode"""
    print()
    print_header("INTERACTIVE MODE")
    print("Type 'quit' to exit")
    print("Type 'batch <path>' to predict batch of images")
    print("Or enter image path for single prediction")
    print("=" * 70)
    
    predictor = Predictor()
    
    if not predictor.model_manager.is_loaded():
        print("❌ Error: Model not loaded.")
        return
    
    while True:
        try:
            user_input = input("\nEnter image path: ").strip()
            
            if user_input.lower() == 'quit':
                print("Exiting...")
                break
            
            if user_input.lower().startswith('batch '):
                path = user_input[6:].strip()
                cmd_batch(path)
            else:
                cmd_predict(user_input)
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


def print_usage():
    """Print usage information"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║         PLANT DISEASE CLASSIFIER - Usage Guide                    ║
╚════════════════════════════════════════════════════════════════════╝

Usage: python main.py <command> [arguments]

Commands:
  predict <image_path>      Predict disease on single image
  batch <folder_path>       Predict on all images in folder
  interactive               Interactive prediction mode
  info                      Show model information

Examples:
  python main.py predict ./uploads/plant.jpg
  python main.py batch ./uploads
  python main.py interactive
  python main.py info

For more help, check README.md
    """)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == "predict" and len(sys.argv) >= 3:
        cmd_predict(sys.argv[2])
    elif command == "batch" and len(sys.argv) >= 3:
        cmd_batch(sys.argv[2])
    elif command == "interactive":
        cmd_interactive()
    elif command == "info":
        cmd_info()
    else:
        print(f"❌ Unknown command: {command}")
        print_usage()


if __name__ == "__main__":
    main()
