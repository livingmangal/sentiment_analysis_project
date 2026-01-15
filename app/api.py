from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from src.predict import initialize_predictor, predict_sentiment
from src.database import init_db, get_db_session, Prediction
import logging
from typing import Dict, Any
import os
import json
import time
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the directory where this file is located (app/)
basedir = os.path.abspath(os.path.dirname(__file__))

# Get static URL path from environment or default to /static
static_url_path = os.environ.get('STATIC_URL_PATH', '/static')

# Use absolute paths to ensure Flask finds static and template folders
app = Flask(__name__, 
    template_folder=os.path.join(basedir, 'templates'),
    static_folder=os.path.join(basedir, 'static'),
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
             "allow_headers": ["Content-Type", "Authorization", "Accept", "X-Session-ID"]
         }
     })

# Initialize the predictor
try:
    initialize_predictor()
    logger.info("Sentiment predictor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize predictor: {str(e)}")
    raise

# Initialize the database
try:
    db_engine = init_db()
    logger.info("Database initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize database: {str(e)}")
    raise


def get_session_id():
    """
    Get or generate a session ID from request headers or cookies.
    Returns a session ID string.
    """
    # Try to get session ID from custom header
    session_id = request.headers.get('X-Session-ID')
    
    # If not in header, try cookies
    if not session_id:
        session_id = request.cookies.get('session_id')
    
    # If still no session ID, generate a new one
    if not session_id:
        session_id = str(uuid.uuid4())
        logger.info(f"Generated new session ID: {session_id}")
    
    return session_id

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
        "raw_score": raw model output,
        "model_info": {                    <--- Added documentation for new field
            "model_version": "v1.0.0",
            "training_date": "2026-01-13",
            ...
        }
    }
    """
    # Handle preflight request
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    time.sleep(3)
        
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
            
            # Get session ID
            session_id = get_session_id()
            prediction_id = None
            
            # Save prediction to database
            try:
                db_session = get_db_session(db_engine)
                prediction = Prediction(
                    input_data=json.dumps({'text': text}),
                    prediction_result=json.dumps(result),
                    session_id=session_id
                )
                db_session.add(prediction)
                db_session.commit()
                prediction_id = prediction.id
                logger.info(f"Saved prediction with ID {prediction_id} for session {session_id}")
                db_session.close()
            except Exception as db_error:
                logger.error(f"Failed to save prediction to database: {str(db_error)}")
                # Continue even if database save fails
            
            # Add session ID to response
            response_data = result.copy()
            response_data['session_id'] = session_id
            response_data['prediction_id'] = prediction_id
                
            return jsonify(response_data), 200
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/history', methods=['GET', 'OPTIONS'])
def get_history() -> tuple[Dict[str, Any], int]:
    """
    Get prediction history for the current session
    
    Returns:
    {
        "predictions": [
            {
                "id": int,
                "input_data": {...},
                "prediction_result": {...},
                "is_favorite": bool,
                "timestamp": "ISO datetime string",
                "session_id": "string"
            },
            ...
        ]
    }
    """
    # Handle preflight request
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        session_id = get_session_id()
        logger.info(f"Fetching history for session: {session_id}")
        
        db_session = get_db_session(db_engine)
        
        # Get predictions for this session, ordered by timestamp descending
        predictions = db_session.query(Prediction).filter(
            Prediction.session_id == session_id
        ).order_by(Prediction.timestamp.desc()).all()
        
        # Convert to dictionaries
        predictions_list = [pred.to_dict() for pred in predictions]
        
        db_session.close()
        
        logger.info(f"Found {len(predictions_list)} predictions for session {session_id}")
        
        return jsonify({
            'predictions': predictions_list,
            'session_id': session_id
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        return jsonify({'error': f'Failed to fetch history: {str(e)}'}), 500


@app.route('/favorite/<int:prediction_id>', methods=['POST', 'OPTIONS'])
def toggle_favorite(prediction_id: int) -> tuple[Dict[str, Any], int]:
    """
    Toggle the favorite status of a prediction
    
    Returns:
    {
        "id": int,
        "is_favorite": bool,
        "message": "string"
    }
    """
    # Handle preflight request
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    try:
        session_id = get_session_id()
        logger.info(f"Toggling favorite for prediction {prediction_id} in session {session_id}")
        
        db_session = get_db_session(db_engine)
        
        # Find the prediction
        prediction = db_session.query(Prediction).filter(
            Prediction.id == prediction_id,
            Prediction.session_id == session_id
        ).first()
        
        if not prediction:
            db_session.close()
            return jsonify({'error': 'Prediction not found'}), 404
        
        # Toggle favorite status
        prediction.is_favorite = not prediction.is_favorite
        db_session.commit()
        
        is_favorite = prediction.is_favorite
        db_session.close()
        
        logger.info(f"Prediction {prediction_id} favorite status set to {is_favorite}")
        
        return jsonify({
            'id': prediction_id,
            'is_favorite': is_favorite,
            'message': 'Favorite status updated successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error toggling favorite: {str(e)}")
        return jsonify({'error': f'Failed to update favorite status: {str(e)}'}), 500
