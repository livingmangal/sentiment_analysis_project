from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from src.predict import initialize_predictor, predict_sentiment
import logging
from typing import Dict, Any
import os
import json
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the directory where this file is located
basedir = os.path.abspath(os.path.dirname(__file__))

# Initialize Flask
# Flask will automatically serve files from the 'static' folder found at 'basedir/static'
app = Flask(__name__, 
    template_folder=os.path.join(basedir, 'templates'),
    static_folder=os.path.join(basedir, 'static')
)

# Disable caching for static files (helps during development)
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
    # In production, you might not want to raise here to keep the server alive
    # raise 

@app.route('/')
def home() -> str:
    """Serve the main page"""
    return render_template('index.html')

# --- REMOVED THE CUSTOM STATIC ROUTE HERE ---
# Flask handles /static/ automatically because we defined static_folder above.

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict() -> tuple[Dict[str, Any], int]:
    # Handle preflight request
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    time.sleep(3) # Simulating delay
        
    try:
        request_data = request.get_data()
        logger.info(f"Received request: {request_data}")
        
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        try:
            data = request.get_json()
        except json.JSONDecodeError as e:
            return jsonify({'error': 'Invalid JSON in request body'}), 400
            
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request'}), 400
        
        text = data['text']
        if not isinstance(text, str) or not text.strip():
            return jsonify({'error': 'Text must be a non-empty string'}), 400
        
        if len(text) > 1000:
            return jsonify({'error': 'Text exceeds the maximum limit of 1000 characters'}), 400
        
        try:
            result = predict_sentiment(text)
            
            # Ensure result is JSON serializable
            if not isinstance(result, dict):
                return jsonify({'error': 'Invalid prediction result format'}), 500
                
            return jsonify(result), 200
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)