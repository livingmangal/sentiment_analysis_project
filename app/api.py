from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from src.predict import initialize_predictor, predict_sentiment
import logging
from typing import Dict, Any
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get static URL path from environment or default to /static
static_url_path = os.environ.get('STATIC_URL_PATH', '/static')

app = Flask(__name__, 
    template_folder='templates',
    static_folder='static',
    static_url_path=static_url_path
)

# Configure app
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Configure CORS
CORS(app, 
     resources={
         r"/*": {
             "origins": "*",
             "methods": ["GET", "POST", "OPTIONS"],
             "allow_headers": ["Content-Type", "Authorization", "Accept"]
         }
     })

# Initialize the predictor
try:
    initialize_predictor()
    logger.info("Sentiment predictor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize predictor: {str(e)}")
    raise

@app.route('/')
def home() -> str:
    """Serve the main page"""
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory(app.static_folder, filename)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict() -> tuple[Dict[str, Any], int]:
    """
    Predict sentiment for given text
    
    Request body:
    {
        "text": "Text to analyze"
    }
    
    Returns:
    {
        "sentiment": "Positive" or "Negative",
        "confidence": float between 0 and 1,
        "raw_score": raw model output
    }
    """
    # Handle preflight request
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
        
    try:
        # Log the incoming request
        request_data = request.get_data()
        logger.info(f"Received request: {request_data}")
        
        # Validate request
        if not request.is_json:
            logger.error("Request is not JSON")
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        try:
            data = request.get_json()
            logger.info(f"Parsed JSON data: {data}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            return jsonify({'error': 'Invalid JSON in request body'}), 400
            
        if not data or 'text' not in data:
            logger.error("Missing text field in request")
            return jsonify({'error': 'Missing "text" field in request'}), 400
        
        text = data['text']
        if not isinstance(text, str) or not text.strip():
            logger.error("Invalid text input")
            return jsonify({'error': 'Text must be a non-empty string'}), 400
        
        # Check text length limit
        if len(text) > 1000:
            logger.error(f"Text length exceeds limit: {len(text)} characters")
            return jsonify({'error': 'Text exceeds the maximum limit of 1000 characters'}), 400
        
        # Get prediction
        try:
            result = predict_sentiment(text)
            logger.info(f"Predicted sentiment for text: {text[:50]}...")
            logger.info(f"Prediction result: {result}")
            
            # Ensure result is JSON serializable
            if not isinstance(result, dict):
                result = {'error': 'Invalid prediction result format'}
                return jsonify(result), 500
                
            return jsonify(result), 200
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500
