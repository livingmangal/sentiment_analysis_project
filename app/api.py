from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from src.predict import initialize_predictor, predict_sentiment
from app.database import init_db, save_prediction, get_all_predictions, clear_all_predictions
from app.analytics import get_sentiment_trends, get_sentiment_summary
import logging
from typing import Dict, Any
import os
import json
import time
import csv
import io

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

# Initialize the database and predictor
try:
    init_db()
    logger.info("Database initialized successfully")
    initialize_predictor()
    logger.info("Sentiment predictor initialized successfully")
except Exception as e:
    logger.error(f"Initialization error: {str(e)}")

@app.route('/')
def home() -> str:
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict() -> tuple[Dict[str, Any], int]:
    # Handle preflight request
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    # Reduced delay for better UX
    time.sleep(1) 
        
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request'}), 400
        
        text = data['text']
        timestamp = data.get('timestamp') # Get timestamp from client
        if not isinstance(text, str) or not text.strip():
            return jsonify({'error': 'Text must be a non-empty string'}), 400
        
        if len(text) > 1000:
            return jsonify({'error': 'Text exceeds the maximum limit of 1000 characters'}), 400
        
        try:
            result = predict_sentiment(text)
            
            if not isinstance(result, dict):
                return jsonify({'error': 'Invalid prediction result format'}), 500
            
            # Save to database with client timestamp if provided
            save_prediction(
                text=text,
                sentiment=result.get('sentiment', 'unknown'),
                confidence=result.get('confidence', 0.0),
                timestamp=timestamp
            )
                
            return jsonify(result), 200
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analytics', methods=['GET'])
def analytics():
    """Endpoint for sentiment trend data"""
    try:
        trends = get_sentiment_trends()
        summary = get_sentiment_summary()
        return jsonify({
            'trends': trends,
            'summary': summary
        }), 200
    except Exception as e:
        logger.error(f"Analytics error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/export', methods=['GET'])
def export_data():
    """Endpoint to export prediction history as CSV"""
    try:
        predictions = get_all_predictions()
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        if predictions:
            writer.writerow(predictions[0].keys())
            # Data
            for pred in predictions:
                writer.writerow(pred.values())
        
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=sentiment_history.csv"}
        )
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear-history', methods=['POST'])
def clear_history():
    """Endpoint to clear prediction history"""
    try:
        clear_all_predictions()
        return jsonify({'status': 'success', 'message': 'History cleared'}), 200
    except Exception as e:
        logger.error(f"Clear history error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)