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

from src.predict import initialize_predictor, predict_sentiment, predict_sentiment_batch
from src.database import init_db, get_db_session, Prediction
from app.database import (
    init_db as init_sqlite_db,
    save_prediction,
    get_all_predictions,
    clear_all_predictions
)
from app.analytics import get_sentiment_trends, get_sentiment_summary

# -------------------------------------------------
# Flask App Initialization (ONLY ONCE)
# -------------------------------------------------
app = Flask(
    __name__,
    template_folder=os.path.join(basedir, "templates"),
    static_folder=os.path.join(basedir, "static")
)

# -------------------------------------------------
# Helper: Standard Error Response
# -------------------------------------------------
def error_response(code, message, suggestion=None, status=400):
    return jsonify({
        "error": {
            "code": code,
            "message": message,
            "suggestion": suggestion
        }
    }), status

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Rate Limiting
# -------------------------------------------------
def get_client_identifier():
    return (
        request.headers.get("X-Session-ID")
        or request.cookies.get("session_id")
        or get_remote_address()
    )

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

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Rate limit exceeded",
        "message": str(e.description),
        "status": 429
    }), 429

# -------------------------------------------------
# App Config
# -------------------------------------------------
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["TEMPLATES_AUTO_RELOAD"] = True

# -------------------------------------------------
# CORS
# -------------------------------------------------
allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
allowed_origins = (
    [o.strip() for o in allowed_origins_env.split(",")]
    if allowed_origins_env != "*"
    else "*"
)

CORS(
    app,
    resources={
        r"/*": {
            "origins": allowed_origins,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": [
                "Content-Type",
                "Authorization",
                "Accept",
                "X-Session-ID"
            ]
        }
    }
)

# -------------------------------------------------
# Initialize Databases & Predictor
# -------------------------------------------------
try:
    init_sqlite_db()
    initialize_predictor()
    logger.info("Initialization successful")
except Exception as e:
    logger.error(f"Initialization error: {str(e)}")

db_engine = init_db()

# -------------------------------------------------
# Session Helper
# -------------------------------------------------
def get_session_id():
    session_id = request.headers.get("X-Session-ID") or request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
    return session_id

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/")
@limiter.exempt
def home():
    return render_template("index.html")

# ---------- ✅ FIXED UPLOAD ROUTE ----------
@app.route("/upload", methods=["POST"])
def upload_csv():
    try:
        file = request.files.get("file")

        if not file:
            return error_response(
                code="NO_FILE",
                message="No file was uploaded.",
                suggestion="Please select a CSV file and try again."
            )

        if not file.filename.lower().endswith(".csv"):
            return error_response(
                code="INVALID_FILE_TYPE",
                message="Unsupported file format.",
                suggestion="Only CSV files are allowed."
            )

        return jsonify({"success": True}), 200

    except Exception as e:
        logger.error(str(e))
        return error_response(
            code="SERVER_ERROR",
            message="Something went wrong on the server.",
            suggestion="Please try again later.",
            status=500
        )

# -------------------------------------------------
@app.route("/predict", methods=["POST", "OPTIONS"])
@limiter.limit("15 per minute")
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        if not request.is_json:
            return error_response(
                "INVALID_CONTENT_TYPE",
                "Invalid request format.",
                "Send JSON with Content-Type application/json."
            )

        data = request.get_json()
        text = data.get("text")

        if not text or not isinstance(text, str):
            return error_response(
                "INVALID_TEXT",
                "Text must be a non-empty string.",
                "Provide valid text for prediction."
            )

        if len(text) > 1000:
            return error_response(
                "TEXT_TOO_LONG",
                "Text exceeds 1000 characters.",
                "Please shorten your text."
            )

        result = predict_sentiment(text)
        session_id = get_session_id()

        save_prediction(
            text=text,
            sentiment=result.get("sentiment", "unknown"),
            confidence=result.get("confidence", 0.0)
        )

        return jsonify({
            **result,
            "session_id": session_id
        }), 200

    except Exception as e:
        logger.error(str(e))
        return error_response(
            "PREDICTION_FAILED",
            "Prediction failed.",
            "Please try again later.",
            500
        )

# -------------------------------------------------
@app.route("/analytics", methods=["GET"])
def analytics():
    try:
        return jsonify({
            "trends": get_sentiment_trends(),
            "summary": get_sentiment_summary()
        }), 200
    except Exception as e:
        logger.error(str(e))
        return error_response(
            "ANALYTICS_ERROR",
            "Failed to load analytics.",
            "Try again later.",
            500
        )

# -------------------------------------------------
@app.route("/export", methods=["GET"])
def export_data():
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
            headers={"Content-Disposition": "attachment; filename=sentiment_history.csv"}
        )
    except Exception as e:
        logger.error(str(e))
        return error_response(
            "EXPORT_ERROR",
            "Failed to export data.",
            "Try again later.",
            500
        )

# -------------------------------------------------
@app.route("/clear-history", methods=["POST"])
def clear_history():
    try:
        clear_all_predictions()
        return jsonify({
            "status": "success",
            "message": "History cleared successfully"
        }), 200
    except Exception as e:
        logger.error(str(e))
        return error_response(
            "CLEAR_HISTORY_ERROR",
            "Failed to clear history.",
            "Try again later.",
            500
        )

# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
