import sys
import os

# Add project root to sys.path automatically
basedir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.dirname(basedir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from src.predict import initialize_predictor, predict_sentiment, predict_sentiment_batch
from src.predict_multilingual import (
    initialize_multilingual_predictor,
    predict_multilingual,
    predict_multilingual_batch,
    _multilingual_predictor
)
from src.language_detection import AdvancedLanguageDetector
from src.database import (
    init_db,
    get_db_session,
    Prediction
)
from app.database import (
    init_db as init_sqlite_db,
    save_prediction,
    get_all_predictions,
    clear_all_predictions
)
from app.analytics import get_sentiment_trends, get_sentiment_summary
import logging
from typing import Dict, Any
import json
import time
import uuid
import csv
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
basedir = os.path.abspath(os.path.dirname(__file__))

from app.admin_routes import admin_bp
from src.ab_testing.framework import ABTestingFramework, TrafficSplitStrategy

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

# Initialize the database and predictor
ab_framework = None

try:
    init_sqlite_db()
    logger.info("SQLite database initialized successfully")
    initialize_predictor()
    logger.info("Standard sentiment predictor initialized successfully")
    
    # Initialize multilingual predictor (models may not exist yet)
    try:
        initialize_multilingual_predictor(auto_detect=True, quantize=True)
        logger.info("Multilingual predictor initialized successfully")
    except Exception as ml_error:
        logger.warning(f"Multilingual predictor not fully initialized: {str(ml_error)}")
    
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

except Exception as e:
    logger.error(f"Initialization error: {str(e)}")

# Initialize SQLAlchemy database
try:
    db_engine = init_db()
    logger.info("SQLAlchemy database initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize database: {str(e)}")
    raise


def get_session_id():
    """Get or generate a session ID from request headers or cookies."""
    session_id = request.headers.get('X-Session-ID')
    
    if not session_id:
        session_id = request.cookies.get('session_id')
    
    if not session_id:
        session_id = str(uuid.uuid4())
        logger.info(f"Generated new session ID: {session_id}")
    
    return session_id


@app.route('/')
@limiter.exempt
def home() -> str:
    """Serve the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'OPTIONS'])
@limiter.limit("15 per minute")
def predict() -> tuple[Dict[str, Any], int]:
    """Standard prediction endpoint - handles English text primarily"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
        
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request'}), 400
        
        text = data['text']
        timestamp = data.get('timestamp')
        
        if not isinstance(text, str) or not text.strip():
            return jsonify({'error': 'Text must be a non-empty string'}), 400
        
        if len(text) > 1000:
            return jsonify({'error': 'Text exceeds the maximum limit of 1000 characters'}), 400
        
        try:
            session_id = get_session_id()
            
            # Use AB testing framework if configured
            if ab_framework and ab_framework.variants:
                try:
                    result = ab_framework.predict(text, session_id=session_id)
                except Exception as e:
                    logger.error(f"AB Framework prediction failed: {e}")
                    result = predict_sentiment(text)
            else:
                result = predict_sentiment(text)
            
            if not isinstance(result, dict):
                result = {'error': 'Invalid prediction result format'}
                return jsonify(result), 500
            
            prediction_id = None
            
            # Save to SQLAlchemy database
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
                logger.error(f"Failed to save prediction to SQLAlchemy database: {str(db_error)}")
            
            # Save to SQLite database for analytics
            try:
                save_prediction(
                    text=text,
                    sentiment=result.get('sentiment', 'unknown'),
                    confidence=result.get('confidence', 0.0),
                    timestamp=timestamp
                )
            except Exception as sqlite_error:
                logger.error(f"Failed to save to SQLite database: {str(sqlite_error)}")
            
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


@app.route('/predict/batch', methods=['POST', 'OPTIONS'])
@limiter.limit("15 per minute")
def predict_batch() -> tuple[Dict[str, Any], int]:
    """Batch prediction endpoint"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
        
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        if not data or 'texts' not in data or not isinstance(data['texts'], list):
            return jsonify({'error': 'Missing or invalid "texts" field in request'}), 400
        
        texts = data['texts']
        if not texts:
            return jsonify({'results': []}), 200
            
        if len(texts) > 50:
            return jsonify({'error': 'Batch size exceeds the maximum limit of 50'}), 400
            
        try:
            session_id = get_session_id()
            
            if ab_framework and ab_framework.variants:
                try:
                    results = ab_framework.predict_batch(texts, session_id=session_id)
                except Exception as e:
                    logger.error(f"AB Framework batch prediction failed: {e}")
                    results = predict_sentiment_batch(texts)
            else:
                results = predict_sentiment_batch(texts)
            
            # Save all predictions
            try:
                db_session = get_db_session(db_engine)
                for text, result in zip(texts, results):
                    prediction = Prediction(
                        input_data=json.dumps({'text': text}),
                        prediction_result=json.dumps(result),
                        session_id=session_id
                    )
                    db_session.add(prediction)
                    
                    # Also save to analytics database
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
                logger.error(f"Failed to save batch predictions: {str(db_error)}")
                
            return jsonify({
                'results': results,
                'session_id': session_id
            }), 200
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Error processing batch request: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict/multilingual', methods=['POST', 'OPTIONS'])
@limiter.limit("30 per minute")
def predict_sentiment_multilingual():
    """Multilingual sentiment prediction endpoint"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
        
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({"error": "Text field is required"}), 400
        
        text = data.get('text', '').strip()
        timestamp = data.get('timestamp')
        
        if not text:
            return jsonify({"error": "Input text cannot be empty"}), 400
        
        language = data.get('language', None)
        
        # Validate language if provided
        if language and language not in ['auto', 'en', 'es', 'fr', 'de', 'hi']:
            return jsonify({
                "error": f"Unsupported language: {language}",
                "supported_languages": ["auto", "en", "es", "fr", "de", "hi"]
            }), 400
        
        # Convert 'auto' to None for auto-detection
        if language == 'auto':
            language = None
        
        session_id = get_session_id()
        
        # Try multilingual prediction, fallback to original if no models available
        try:
            result = predict_multilingual(text, language=language)
        except ValueError as ve:
            logger.warning(f"Multilingual prediction failed, using fallback: {str(ve)}")
            result = predict_sentiment(text)
            result['language'] = 'en'
            result['fallback'] = True
        
        # Save to databases
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
            result['prediction_id'] = prediction_id
            db_session.close()
        except Exception as db_error:
            logger.error(f"Database save error: {str(db_error)}")
        
        try:
            save_prediction(
                text=text,
                sentiment=result.get('sentiment', 'unknown'),
                confidence=result.get('confidence', 0.0),
                timestamp=timestamp
            )
        except Exception as sqlite_error:
            logger.error(f"SQLite save error: {str(sqlite_error)}")
        
        result['session_id'] = session_id
        return jsonify(result), 200
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error in multilingual prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/predict/multilingual/batch', methods=['POST', 'OPTIONS'])
@limiter.limit("10 per minute")
def predict_sentiment_multilingual_batch():
    """Batch multilingual sentiment prediction endpoint"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
        
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        
        if 'texts' not in data:
            return jsonify({"error": "Texts field is required"}), 400
        
        texts = data.get('texts', [])
        
        if not isinstance(texts, list):
            return jsonify({"error": "Texts must be a list"}), 400
        
        if len(texts) == 0:
            return jsonify({"error": "Texts list cannot be empty"}), 400
        
        if len(texts) > 50:
            return jsonify({"error": "Maximum 50 texts per batch"}), 400
        
        texts = [text.strip() for text in texts if text and text.strip()]
        
        if len(texts) == 0:
            return jsonify({"error": "All texts are empty"}), 400
        
        language = data.get('language', None)
        
        if language == 'auto':
            language = None
        
        session_id = get_session_id()
        
        # Try multilingual batch prediction, fallback if needed
        try:
            results = predict_multilingual_batch(texts, language=language)
        except ValueError:
            results = predict_sentiment_batch(texts)
            for r in results:
                r['language'] = 'en'
                r['fallback'] = True
        
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
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error in batch multilingual prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/languages', methods=['GET'])
@limiter.exempt
def get_available_languages():
    """Get list of available languages"""
    try:
        available = set()
        if _multilingual_predictor:
            available = set(_multilingual_predictor.get_available_languages())
        
        # Always show English as available since we have fallback
        available.add('en')
        
        languages = [
            {"code": "en", "name": "English", "flag": "🇬🇧", "available": "en" in available},
            {"code": "es", "name": "Spanish", "flag": "🇪🇸", "available": "es" in available},
            {"code": "fr", "name": "French", "flag": "🇫🇷", "available": "fr" in available},
            {"code": "de", "name": "German", "flag": "🇩🇪", "available": "de" in available},
            {"code": "hi", "name": "Hindi", "flag": "🇮🇳", "available": "hi" in available}
        ]
        
        return jsonify({
            "languages": languages,
            "auto_detect_available": True
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting available languages: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/detect-language', methods=['POST'])
@limiter.limit("60 per minute")
def detect_language():
    """Language detection endpoint"""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({"error": "Text field is required"}), 400
        
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({"error": "Input text cannot be empty"}), 400
        
        # Use the language detector
        if _multilingual_predictor:
            detected = _multilingual_predictor.language_detector.detect(text)
            confidence = _multilingual_predictor.language_detector.get_confidence(text)
        else:
            detector = AdvancedLanguageDetector()
            detected = detector.detect(text)
            confidence = detector.get_confidence(text)
        
        return jsonify({
            "detected_language": detected,
            "confidence": confidence
        }), 200
        
    except Exception as e:
        logger.error(f"Error in language detection: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/history', methods=['GET', 'OPTIONS'])
def get_history() -> tuple[Dict[str, Any], int]:
    """Get prediction history for current session"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        session_id = get_session_id()
        logger.info(f"Fetching history for session: {session_id}")

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


@app.route('/analytics', methods=['GET'])
def analytics():
    """Get analytics data including trends and summary"""
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
        # Clear SQLite history
        clear_all_predictions()
        
        # Clear SQLAlchemy history for current session
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