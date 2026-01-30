import sys
import os
import logging
import json
import time
import uuid
import csv
import io
from typing import Dict, Any

# Add project root to sys.path automatically
basedir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.dirname(basedir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Model imports
from src.predict import initialize_predictor, predict_sentiment, predict_sentiment_batch

# Database imports
from src.database import (
    init_db,
    get_db_session,
    Prediction,
    Feedback
)
from app.database import (
    init_db as init_sqlite_db,
    save_prediction,
    get_all_predictions,
    clear_all_predictions
)
from app.analytics import get_sentiment_trends, get_sentiment_summary

# Analytics and AB Testing
from src.analytics import init_analytics_db, log_request, get_analytics_data
from app.admin_routes import admin_bp
from src.ab_testing.framework import ABTestingFramework, TrafficSplitStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask
app = Flask(__name__, 
    template_folder=os.path.join(basedir, 'templates'),
    static_folder=os.path.join(basedir, 'static')
)

# Register Admin Blueprint
app.register_blueprint(admin_bp)

def get_client_identifier():
    """
    Returns a unique identifier for the client based on:
    1. X-Session-ID header
    2. session_id cookie
    3. Remote IP address (fallback)
    """
    return request.headers.get('X-Session-ID') or \
           request.cookies.get('session_id') or \
           get_remote_address()

# Initialize Rate Limiter
limiter = Limiter(
    key_func=get_client_identifier,
    app=app,
    default_limits=["200 per day", "50 per hour", "15 per minute"],
    storage_uri="memory://",
    strategy="fixed-window",
    headers_enabled=True
)

@limiter.request_filter
def exempt_options():
    return request.method == "OPTIONS"

# Custom error message for rate limiting
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Rate limit exceeded",
        "message": str(e.description),
        "status": 429
    }), 429

# Disable caching for static files
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Configure CORS
allowed_origins_env = os.environ.get('ALLOWED_ORIGINS', '*')
if allowed_origins_env != '*':
    allowed_origins = [origin.strip() for origin in allowed_origins_env.split(',')]
else:
    allowed_origins = '*'

CORS(app, 
     resources={
         r"/*": {
             "origins": allowed_origins,
             "methods": ["GET", "POST", "OPTIONS"],
             "allow_headers": ["Content-Type", "Authorization", "Accept", "X-Session-ID"]
         }
     })

# Initialize the database and frameworks
ab_framework = None
db_engine = None

try:
    init_sqlite_db()
    init_analytics_db()
    logger.info("Local databases initialized successfully")
    
    # Initialize AB Testing Framework
    try:
        ab_framework = ABTestingFramework.load_config("sentiment_v1")
        logger.info("AB Testing Framework loaded from config")
    except Exception:
        logger.info("Initializing new AB Testing Framework")
        ab_framework = ABTestingFramework("sentiment_v1", strategy=TrafficSplitStrategy.SESSION_HASH)
        from src.registry import ModelRegistry
        registry = ModelRegistry()
        latest = registry.get_latest_active_model()
        if latest:
             from src.ab_testing.framework import ModelVariant
             variant = ModelVariant(
                 variant_id=latest.version,
                 model_path=latest.model_path,
                 preprocessor_path=latest.preprocessor_path,
                 description="Default active model"
             )
             ab_framework.add_variant(variant)
             ab_framework.save_config()

    # Initialize SQLAlchemy database
    db_engine = init_db()
    logger.info("SQLAlchemy database initialized successfully")
    
    # Initialize predictor
    initialize_predictor()
    logger.info("Predictor initialized successfully")

except Exception as e:
    logger.error(f"Initialization error: {str(e)}")

def get_session_id():
    """Get or generate session ID"""
    session_id = request.headers.get('X-Session-ID')
    if not session_id:
        session_id = request.cookies.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
    return session_id

@app.route('/')
@limiter.exempt
def home() -> str:
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'OPTIONS'])
@limiter.limit("15 per minute")
def predict() -> tuple[Dict[str, Any], int]:
    """Prediction endpoint with AB testing support"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
        
    start_time = time.time()
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request'}), 400
        
        text = data['text']
        session_id = get_session_id()
        
        # Log request to analytics
        log_request(
            endpoint='/predict',
            method='POST',
            status_code=200,
            response_time=0, # Will update later
            session_id=session_id
        )

        if not isinstance(text, str) or not text.strip():
            return jsonify({'error': 'Text must be a non-empty string'}), 400
        
        if len(text) > 1000:
            return jsonify({'error': 'Text exceeds the maximum limit of 1000 characters'}), 400
        
        # Prediction with AB Testing fallback
        try:
            # 1. Try AB testing if enabled
            if ab_framework and ab_framework.variants:
                try:
                    result = ab_framework.predict(text, session_id=session_id)
                    result['ab_test'] = True
                except Exception as e:
                    logger.error(f"AB Testing failed: {e}")
                    result = predict_sentiment(text)
            else:
                # 2. Standard prediction
                result = predict_sentiment(text)
            
            # 3. Add timing and metadata
            result['response_time'] = round(time.time() - start_time, 4)
            result['session_id'] = session_id
            
            # Save to SQLAlchemy DB
            try:
                db_session = get_db_session(db_engine)
                prediction = Prediction(
                    input_data=json.dumps({'text': text}),
                    prediction_result=json.dumps(result),
                    session_id=session_id
                )
                db_session.add(prediction)
                db_session.commit()
                result['prediction_id'] = prediction.id
                db_session.close()
            except Exception as db_error:
                logger.error(f"SQLAlchemy save error: {str(db_error)}")

            # Save to SQLite Analytics
            try:
                save_prediction(
                    text=text,
                    sentiment=result.get('sentiment', 'unknown'),
                    confidence=result.get('confidence', 0.0)
                )
            except Exception as sqlite_error:
                logger.error(f"SQLite save error: {str(sqlite_error)}")

            return jsonify(result), 200
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST', 'OPTIONS'])
@limiter.limit("15 per minute")
def predict_batch() -> tuple[Dict[str, Any], int]:
    """Batch prediction"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
        
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        if not data or 'texts' not in data or not isinstance(data['texts'], list):
            return jsonify({'error': 'Missing or invalid "texts" field'}), 400
        
        texts = data['texts']
        if not texts:
            return jsonify({'results': []}), 200
            
        if len(texts) > 50:
            return jsonify({'error': 'Batch size exceeds limit of 50'}), 400
            
        session_id = get_session_id()
        
        try:
            results = predict_sentiment_batch(texts)
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500
        
        # Save predictions
        try:
            db_session = get_db_session(db_engine)
            for text, result in zip(texts, results):
                prediction = Prediction(
                    input_data=json.dumps({'text': text}),
                    prediction_result=json.dumps(result),
                    session_id=session_id
                )
                db_session.add(prediction)
                
                try:
                    save_prediction(
                        text=text,
                        sentiment=result.get('sentiment', 'unknown'),
                        confidence=result.get('confidence', 0.0)
                    )
                except Exception as sqlite_error:
                    logger.error(f"SQLite save error: {str(sqlite_error)}")
            
            db_session.commit()
            db_session.close()
        except Exception as db_error:
            logger.error(f"Database save error: {str(db_error)}")
        
        return jsonify({
            "results": results,
            "total_count": len(results),
            "session_id": session_id
        }), 200
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/history', methods=['GET', 'OPTIONS'])
def get_history() -> tuple[Dict[str, Any], int]:
    """Get prediction history for current session"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        session_id = get_session_id()
        db_session = get_db_session(db_engine)
        predictions = db_session.query(Prediction).filter(
            Prediction.session_id == session_id
        ).order_by(Prediction.timestamp.desc()).all()
        predictions_list = [pred.to_dict() for pred in predictions]
        db_session.close()
        return jsonify({
            'predictions': predictions_list,
            'session_id': session_id
        }), 200
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/favorite/<int:prediction_id>', methods=['POST', 'OPTIONS'])
def toggle_favorite(prediction_id: int) -> tuple[Dict[str, Any], int]:
    """Toggle favorite status of a prediction"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        session_id = get_session_id()
        db_session = get_db_session(db_engine)
        prediction = db_session.query(Prediction).filter(
            Prediction.id == prediction_id,
            Prediction.session_id == session_id
        ).first()

        if not prediction:
            db_session.close()
            return jsonify({'error': 'Prediction not found'}), 404

        prediction.is_favorite = not prediction.is_favorite
        db_session.commit()
        is_favorite = prediction.is_favorite
        db_session.close()
        return jsonify({
            'id': prediction_id,
            'is_favorite': is_favorite,
            'message': 'Favorite status updated successfully'
        }), 200
    except Exception as e:
        logger.error(f"Error toggling favorite: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST', 'OPTIONS'])
@limiter.limit("10 per minute")
def submit_feedback() -> tuple[Dict[str, Any], int]:
    """Submit user feedback for predictions"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400

        data = request.get_json()
        required_fields = ['text', 'actual_sentiment']
        if not all(field in data for field in required_fields):
            return jsonify({'error': f'Missing required fields: {required_fields}'}), 400

        text = data['text']
        actual_sentiment = data['actual_sentiment']
        prediction_id = data.get('prediction_id')
        predicted_sentiment = data.get('predicted_sentiment')

        # Map sentiment labels to indices if they are strings
        sentiment_map = {"Negative": 0, "Positive": 1, "Neutral": 2}
        if isinstance(actual_sentiment, str):
            actual_sentiment = sentiment_map.get(actual_sentiment)
        if isinstance(predicted_sentiment, str):
            predicted_sentiment = sentiment_map.get(predicted_sentiment)

        if actual_sentiment is None:
            return jsonify({'error': 'Invalid actual_sentiment value'}), 400

        db_session = get_db_session(db_engine)
        feedback = Feedback(
            prediction_id=prediction_id,
            text=text,
            predicted_sentiment=predicted_sentiment,
            actual_sentiment=actual_sentiment
        )
        db_session.add(feedback)
        db_session.commit()
        feedback_id = feedback.id
        db_session.close()

        logger.info(f"Received feedback for prediction {prediction_id}: actual={actual_sentiment}")
        return jsonify({
            'status': 'success',
            'feedback_id': feedback_id,
            'message': 'Feedback received. Thank you!'
        }), 200
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analytics', methods=['GET'])
def analytics():
    """Analytics endpoint with admin and public modes"""
    # 1. Admin Security Check
    admin_key = request.headers.get('X-Admin-Key')
    if admin_key and admin_key == 'secret_admin_key_123':
        try:
            stats = get_analytics_data()
            return jsonify(stats), 200
        except Exception as e:
            logger.error(f"Admin analytics error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    # 2. Public Fallback
    try:
        trends = get_sentiment_trends()
        summary = get_sentiment_summary()
        return jsonify({
            'trends': trends,
            'summary': summary
        }), 200
    except Exception as e:
        logger.error(f"Public analytics error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/export', methods=['GET'])
def export_data():
    """Export prediction history as CSV"""
    try:
        predictions = get_all_predictions()
        output = io.StringIO()
        writer = csv.writer(output)
        if predictions:
            writer.writerow(predictions[0].keys())
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
    """Clear all prediction history for current session"""
    try:
        clear_all_predictions()
        session_id = get_session_id()
        db_session = get_db_session(db_engine)
        db_session.query(Prediction).filter(Prediction.session_id == session_id).delete()
        db_session.commit()
        db_session.close()
        return jsonify({'status': 'success', 'message': 'History cleared'}), 200
    except Exception as e:
        logger.error(f"Clear history error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
