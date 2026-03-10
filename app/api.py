"""
Plant Disease Classifier - Flask API (Optional)
This is an example Flask API you can use to expose the classifier as a web service.

Installation:
    pip install flask flask-cors python-dotenv

Usage:
    python api.py
    
Then access:
    GET  /health - Check API status
    POST /predict - Make a prediction
    GET  /info - Model information
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from pathlib import Path
from predictor import Predictor
import logging

# Setup
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize predictor
predictor = Predictor()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif'}
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": predictor.model_manager.is_loaded()
    }), 200


@app.route('/info', methods=['GET'])
def info():
    """Get model information"""
    if not predictor.model_manager.is_loaded():
        return jsonify({"error": "Model not loaded"}), 500
    
    model = predictor.model_manager.get_model()
    classes = predictor.model_manager.get_classes()
    
    return jsonify({
        "status": "success",
        "model": {
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape),
            "total_parameters": int(model.count_params())
        },
        "classes": {
            "total": len(classes),
            "samples": classes[:10]
        }
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a prediction on uploaded image
    
    Usage:
        curl -X POST -F "image=@plant.jpg" http://localhost:5000/predict
    """
    
    # Check if model is loaded
    if not predictor.model_manager.is_loaded():
        return jsonify({"error": "Model not loaded"}), 500
    
    # Check if image in request
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            "error": "File type not allowed",
            "allowed": list(ALLOWED_EXTENSIONS)
        }), 400
    
    try:
        # Save temporary file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        # Make prediction
        result = predictor.predict_single_image(filepath)
        
        if not result.get('success'):
            return jsonify({"error": result.get('error')}), 400
        
        # Format response
        response = {
            "status": "success",
            "prediction": {
                "disease": result['predicted_disease'],
                "confidence": round(result['confidence'], 4),
                "confidence_percentage": result['confidence_str']
            },
            "top_predictions": [
                {
                    "rank": p['rank'],
                    "disease": p['disease'],
                    "confidence": round(p['confidence'], 4),
                    "confidence_percentage": p['confidence_str']
                }
                for p in result['top_predictions']
            ],
            "processing_time_seconds": round(result['processing_time'], 3),
            "image_info": result['image_info']
        }
        
        # Clean up
        os.remove(filepath)
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/batch', methods=['POST'])
def batch_predict():
    """
    Batch predict on multiple images (as list)
    
    Usage:
        curl -X POST -F "images=@image1.jpg" -F "images=@image2.jpg" \\
             http://localhost:5000/batch
    """
    
    if not predictor.model_manager.is_loaded():
        return jsonify({"error": "Model not loaded"}), 500
    
    if 'images' not in request.files:
        return jsonify({"error": "No images part"}), 400
    
    files = request.files.getlist('images')
    
    if len(files) == 0:
        return jsonify({"error": "No files selected"}), 400
    
    results = []
    
    try:
        for file in files:
            if file.filename == '':
                continue
            
            if not allowed_file(file.filename):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "File type not allowed"
                })
                continue
            
            # Save and predict
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            
            pred_result = predictor.predict_single_image(filepath)
            
            if pred_result.get('success'):
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "disease": pred_result['predicted_disease'],
                    "confidence": round(pred_result['confidence'], 4),
                    "confidence_percentage": pred_result['confidence_str']
                })
            else:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": pred_result.get('error')
                })
            
            # Clean up
            os.remove(filepath)
        
        successful = sum(1 for r in results if r.get('success'))
        
        return jsonify({
            "status": "success",
            "total_processed": len(results),
            "successful": successful,
            "results": results
        }), 200
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "status": "error",
        "message": "Endpoint not found",
        "available_endpoints": [
            "GET /health",
            "GET /info",
            "POST /predict",
            "POST /batch"
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "status": "error",
        "message": "Internal server error"
    }), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║  Plant Disease Classifier - Flask API                 ║
    ╚════════════════════════════════════════════════════════╝
    
    📌 Available Endpoints:
    
    GET  http://localhost:5000/health
         Check API status
    
    GET  http://localhost:5000/info
         Get model information
    
    POST http://localhost:5000/predict
         Predict on single image
         Usage: curl -F "image=@photo.jpg" http://localhost:5000/predict
    
    POST http://localhost:5000/batch
         Batch predict on multiple images
         Usage: curl -F "images=@img1.jpg" -F "images=@img2.jpg" \\
                http://localhost:5000/batch
    
    ✅ Starting server on http://localhost:5000
    Press CTRL+C to stop
    """)
    
    # Run Flask app
    app.run(
        debug=False,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )
