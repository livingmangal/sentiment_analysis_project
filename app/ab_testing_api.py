"""
Flask API Integration for A/B Testing
Adds endpoints for managing and monitoring A/B tests
"""
from flask import Blueprint, jsonify, request
from typing import Dict, Any
import logging
import os
import sys

# Add project root to path
basedir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.dirname(basedir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ab_testing.framework import ABTestingFramework, ModelVariant, TrafficSplitStrategy
from src.ab_testing.metrics import MetricsCollector
from src.ab_testing.statistics import StatisticalAnalyzer
from src.ab_testing.dashboard import DashboardGenerator

logger = logging.getLogger(__name__)

# Create Blueprint
ab_testing_bp = Blueprint('ab_testing', __name__, url_prefix='/ab-test')

# Global A/B test instance (will be configured at startup)
active_experiment: ABTestingFramework = None


def initialize_ab_test(
    experiment_name: str,
    variants: list,
    strategy: str = "session_hash"
):
    """
    Initialize A/B testing framework
    
    Args:
        experiment_name: Name of the experiment
        variants: List of variant configurations
        strategy: Traffic splitting strategy
    """
    global active_experiment
    
    try:
        strategy_enum = TrafficSplitStrategy(strategy)
    except ValueError:
        strategy_enum = TrafficSplitStrategy.SESSION_HASH
    
    active_experiment = ABTestingFramework(
        experiment_name=experiment_name,
        strategy=strategy_enum,
        enable_logging=True
    )
    
    # Add variants
    for variant_config in variants:
        variant = ModelVariant(
            variant_id=variant_config["variant_id"],
            model_path=variant_config["model_path"],
            preprocessor_path=variant_config["preprocessor_path"],
            metadata_path=variant_config.get("metadata_path"),
            weight=variant_config.get("weight", 1.0),
            description=variant_config.get("description", "")
        )
        active_experiment.add_variant(variant)
    
    # Save configuration
    active_experiment.save_config()
    
    logger.info(f"Initialized A/B test: {experiment_name} with {len(variants)} variants")


@ab_testing_bp.route('/predict', methods=['POST'])
def ab_predict():
    """
    Make prediction using A/B testing framework
    """
    if active_experiment is None:
        return jsonify({'error': 'No active A/B test'}), 503
    
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field'}), 400
        
        text = data['text']
        session_id = request.headers.get('X-Session-ID') or request.cookies.get('session_id')
        
        # Make prediction with A/B test
        result = active_experiment.predict(
            text=text,
            session_id=session_id,
            metadata=data.get('metadata')
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"A/B test prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@ab_testing_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Get metrics for all variants
    """
    if active_experiment is None:
        return jsonify({'error': 'No active A/B test'}), 503
    
    try:
        collector = MetricsCollector(active_experiment.experiment_name)
        summary = collector.get_summary_stats()
        
        return jsonify(summary), 200
        
    except Exception as e:
        logger.error(f"Error fetching metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500


@ab_testing_bp.route('/metrics/<variant_id>', methods=['GET'])
def get_variant_metrics(variant_id: str):
    """
    Get detailed metrics for a specific variant
    """
    if active_experiment is None:
        return jsonify({'error': 'No active A/B test'}), 503
    
    try:
        collector = MetricsCollector(active_experiment.experiment_name)
        metrics = collector.collect_metrics(variant_id)
        
        # Convert to dict for JSON serialization
        from dataclasses import asdict
        return jsonify(asdict(metrics)), 200
        
    except Exception as e:
        logger.error(f"Error fetching variant metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500


@ab_testing_bp.route('/compare', methods=['POST'])
def compare_variants():
    """
    Compare two variants statistically
    """
    if active_experiment is None:
        return jsonify({'error': 'No active A/B test'}), 503
    
    try:
        data = request.get_json()
        variant_a = data.get('variant_a')
        variant_b = data.get('variant_b')
        
        if not variant_a or not variant_b:
            return jsonify({'error': 'Both variant_a and variant_b required'}), 400
        
        # Collect data for both variants
        collector = MetricsCollector(active_experiment.experiment_name)
        
        # Load prediction logs and extract data
        variant_a_data = _load_variant_data(
            active_experiment.experiment_name,
            variant_a
        )
        variant_b_data = _load_variant_data(
            active_experiment.experiment_name,
            variant_b
        )
        
        # Run statistical analysis
        analyzer = StatisticalAnalyzer()
        results = analyzer.run_complete_analysis(variant_a_data, variant_b_data)
        
        # Convert results to dict
        response = {}
        for test_name, result in results.items():
            from dataclasses import asdict
            response[test_name] = asdict(result)
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error comparing variants: {str(e)}")
        return jsonify({'error': str(e)}), 500


@ab_testing_bp.route('/dashboard', methods=['GET'])
def generate_dashboard():
    """
    Generate and return dashboard HTML
    """
    if active_experiment is None:
        return jsonify({'error': 'No active A/B test'}), 503
    
    try:
        generator = DashboardGenerator(active_experiment.experiment_name)
        dashboard_path = generator.generate_dashboard()
        
        # Read and return HTML
        with open(dashboard_path, 'r') as f:
            html_content = f.read()
        
        from flask import Response
        return Response(html_content, mimetype='text/html')
        
    except Exception as e:
        logger.error(f"Error generating dashboard: {str(e)}")
        return jsonify({'error': str(e)}), 500


@ab_testing_bp.route('/config', methods=['GET'])
def get_config():
    """
    Get current A/B test configuration
    """
    if active_experiment is None:
        return jsonify({'error': 'No active A/B test'}), 503
    
    try:
        config = {
            'experiment_name': active_experiment.experiment_name,
            'strategy': active_experiment.strategy.value,
            'variants': [
                variant.to_dict()
                for variant in active_experiment.variants.values()
            ]
        }
        
        return jsonify(config), 200
        
    except Exception as e:
        logger.error(f"Error getting config: {str(e)}")
        return jsonify({'error': str(e)}), 500


def _load_variant_data(experiment_name: str, variant_id: str) -> Dict[str, Any]:
    """
    Load prediction data for a variant from log files
    
    Returns:
        Dict with lists of confidence scores, inference times, and sentiment counts
    """
    import json
    
    experiment_dir = os.path.join(
        project_root,
        "experiments",
        experiment_name
    )
    
    log_file = os.path.join(experiment_dir, f"{variant_id}_predictions.jsonl")
    
    data = {
        "confidence": [],
        "inference_time": [],
        "sentiment": {"Positive": 0, "Negative": 0}
    }
    
    if not os.path.exists(log_file):
        return data
    
    with open(log_file, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            
            if entry.get("confidence") is not None:
                data["confidence"].append(entry["confidence"])
            
            if entry.get("inference_time_ms") is not None:
                data["inference_time"].append(entry["inference_time_ms"])
            
            sentiment = entry.get("sentiment")
            if sentiment in data["sentiment"]:
                data["sentiment"][sentiment] += 1
    
    return data


# Helper function to register with main Flask app
def register_ab_testing(app, experiment_config: Dict[str, Any] = None):
    """
    Register A/B testing blueprint with Flask app
    
    Args:
        app: Flask application instance
        experiment_config: Configuration for A/B test (optional)
            {
                "experiment_name": "test_v1_vs_v2",
                "strategy": "session_hash",
                "variants": [
                    {
                        "variant_id": "control",
                        "model_path": "models/lstm_model.pth",
                        "preprocessor_path": "models/preprocessor.pkl",
                        "weight": 1.0
                    },
                    {
                        "variant_id": "variant_a",
                        "model_path": "models/lstm_model_v2.pth",
                        "preprocessor_path": "models/preprocessor_v2.pkl",
                        "weight": 1.0
                    }
                ]
            }
    """
    app.register_blueprint(ab_testing_bp)
    
    if experiment_config:
        initialize_ab_test(
            experiment_name=experiment_config["experiment_name"],
            variants=experiment_config["variants"],
            strategy=experiment_config.get("strategy", "session_hash")
        )
        
        logger.info("A/B testing endpoints registered and initialized")