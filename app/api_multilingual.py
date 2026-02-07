"""
Extended Flask API with Multilingual Sentiment Analysis Support
Add these routes to your existing app/api.py or use as separate blueprint
"""
from flask import Blueprint, request, jsonify, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Import multilingual predictor
from src.predict_multilingual import (
    _multilingual_predictor,
    initialize_multilingual_predictor,
    predict_multilingual,
    predict_multilingual_batch,
)

# Load environment variables
load_dotenv()

multilingual_bp = Blueprint('multilingual', __name__)

# Initialize multilingual predictor flag
_predictor_initialized = False


def ensure_predictor_initialized():
    """Ensure the multilingual predictor is initialized (called on first request)"""
    global _predictor_initialized
    if not _predictor_initialized:
        try:
            # Initialize with XLM-RoBERTa
            initialize_multilingual_predictor(
                model_name="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                auto_detect=True, 
                quantize=True
            )
            print("Multilingual predictor initialized successfully")
            _predictor_initialized = True
        except Exception as e:
            print(f"Warning: Could not initialize multilingual predictor: {e}")
            print("Multilingual endpoints may not work properly")
            _predictor_initialized = True  # Prevent repeated initialization attempts


@multilingual_bp.before_request
def before_request():
    """Initialize predictor on first request if needed"""
    ensure_predictor_initialized()


@multilingual_bp.route('/multilingual')
def home():
    """Serve the multilingual UI"""
    return render_template('index_multilingual.html')


@multilingual_bp.route('/predict/multilingual', methods=['POST'])
# Limiter is applied globally in main app or can be applied here if we pass the limiter instance
# For now we'll assume the main app handles rate limits or we skip strict limits on BP for simplicity

def predict_sentiment_multilingual():
    """
    Multilingual sentiment prediction endpoint
    
    Request body:
    {
        "text": "Your text here",
        "language": "en"  // optional, will auto-detect if not provided
    }
    
    Response:
    {
        "sentiment": "Positive/Negative",
        "confidence": 0.95,
        "raw_score": 2.5,
        "language": "en",
        "model_info": {...},
        "inference_time_ms": 12.5
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.get_json()

        # Validate text field
        if 'text' not in data:
            return jsonify({"error": "Text field is required"}), 400

        text = data.get('text', '').strip()

        if not text:
            return jsonify({"error": "Input text cannot be empty"}), 400

        # Optional language parameter
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

        # Make prediction
        result = predict_multilingual(text, language=language)

        return jsonify(result), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"Error in multilingual prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500


@multilingual_bp.route('/predict/multilingual/batch', methods=['POST'])

def predict_sentiment_multilingual_batch():
    """
    Batch multilingual sentiment prediction endpoint
    
    Request body:
    {
        "texts": ["Text 1", "Text 2", ...],
        "language": "en"  // optional, will auto-detect per text if not provided
    }
    
    Response:
    {
        "results": [
            {
                "sentiment": "Positive",
                "confidence": 0.95,
                "language": "en",
                ...
            },
            ...
        ],
        "total_count": 2
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        data = request.get_json()

        # Validate texts field
        if 'texts' not in data:
            return jsonify({"error": "Texts field is required"}), 400

        texts = data.get('texts', [])

        if not isinstance(texts, list):
            return jsonify({"error": "Texts must be a list"}), 400

        if len(texts) == 0:
            return jsonify({"error": "Texts list cannot be empty"}), 400

        if len(texts) > 50:
            return jsonify({"error": "Maximum 50 texts per batch"}), 400

        # Filter out empty texts
        texts = [text.strip() for text in texts if text and text.strip()]

        if len(texts) == 0:
            return jsonify({"error": "All texts are empty"}), 400

        # Optional language parameter
        language = data.get('language', None)

        if language == 'auto':
            language = None

        # Make predictions
        results = predict_multilingual_batch(texts, language=language)

        return jsonify({
            "results": results,
            "total_count": len(results)
        }), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"Error in batch multilingual prediction: {e}")
        return jsonify({"error": "Internal server error"}), 500


@multilingual_bp.route('/languages', methods=['GET'])
def get_available_languages():
    """
    Get list of available languages
    
    Response:
    {
        "languages": [
            {
                "code": "en",
                "name": "English",
                "flag": "🇬🇧",
                "available": true
            },
            ...
        ]
    }
    """
    try:
        # Check which languages have models loaded
        available = set()
        if _multilingual_predictor:
            available = set(_multilingual_predictor.get_available_languages())

        languages = [
            {
                "code": "en",
                "name": "English",
                "flag": "🇬🇧",
                "available": "en" in available
            },
            {
                "code": "es",
                "name": "Spanish",
                "flag": "🇪🇸",
                "available": "es" in available
            },
            {
                "code": "fr",
                "name": "French",
                "flag": "🇫🇷",
                "available": "fr" in available
            },
            {
                "code": "de",
                "name": "German",
                "flag": "🇩🇪",
                "available": "de" in available
            },
            {
                "code": "hi",
                "name": "Hindi",
                "flag": "🇮🇳",
                "available": "hi" in available
            }
        ]

        return jsonify({
            "languages": languages,
            "auto_detect_available": True
        }), 200

    except Exception as e:
        print(f"Error getting available languages: {e}")
        return jsonify({"error": "Internal server error"}), 500


@multilingual_bp.route('/detect-language', methods=['POST'])

def detect_language():
    """
    Language detection endpoint
    
    Request body:
    {
        "text": "Your text here"
    }
    
    Response:
    {
        "detected_language": "en",
        "confidence": {
            "en": 0.95,
            "es": 0.03,
            "fr": 0.01,
            "de": 0.005,
            "hi": 0.005
        }
    }
    """
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
            from src.language_detection import AdvancedLanguageDetector
            detector = AdvancedLanguageDetector()
            detected = detector.detect(text)
            confidence = detector.get_confidence(text)

        return jsonify({
            "detected_language": detected,
            "confidence": confidence
        }), 200

    except Exception as e:
        print(f"Error in language detection: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit exceeded"""
    return jsonify({
        "error": "Rate limit exceeded",
        "message": str(e.description)
    }), 429


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
