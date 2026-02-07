from flask import Blueprint, jsonify, request, render_template, Response
from src.registry import ModelRegistry
from src.ab_testing.framework import ABTestingFramework, ModelVariant, TrafficSplitStrategy
import os
import logging
from src.database import get_db_session
from src.finetune import finetune_model
from app.analytics import get_dashboard_stats, export_predictions_csv
from datetime import datetime

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')
logger = logging.getLogger(__name__)

@admin_bp.route('/')
def admin_dashboard():
    return render_template('admin.html')

@admin_bp.route('/api/stats', methods=['GET'])
def get_stats():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    variant_id = request.args.get('variant_id')
    language = request.args.get('language')
    
    if start_date: start_date = datetime.fromisoformat(start_date)
    if end_date: end_date = datetime.fromisoformat(end_date)
    
    stats = get_dashboard_stats(start_date, end_date, variant_id, language)
    return jsonify(stats)

@admin_bp.route('/api/export/csv', methods=['GET'])
def export_csv():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    if start_date: start_date = datetime.fromisoformat(start_date)
    if end_date: end_date = datetime.fromisoformat(end_date)
    
    csv_data = export_predictions_csv(start_date, end_date)
    
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename=predictions_{datetime.now().strftime('%Y%m%d')}.csv"}
    )

@admin_bp.route('/api/models', methods=['GET'])
def list_models():
    registry = ModelRegistry()
    models = registry.list_models()
    return jsonify([m.to_dict() for m in models])

@admin_bp.route('/api/models/<version>/status', methods=['POST'])
def update_model_status(version):
    data = request.json
    new_status = data.get('status')
    if not new_status:
        return jsonify({'error': 'Status is required'}), 400
    
    registry = ModelRegistry()
    if registry.set_status(version, new_status):
        return jsonify({'success': True, 'message': f'Status updated to {new_status}'})
    return jsonify({'error': 'Model not found'}), 404

@admin_bp.route('/api/experiment', methods=['GET'])
def get_experiment_config():
    # Load current experiment config
    framework = ABTestingFramework("sentiment_v1", enable_logging=False)
    # Reload from config file if exists
    try:
        loaded_framework = ABTestingFramework.load_config("sentiment_v1")
        config = {
            "experiment_name": loaded_framework.experiment_name,
            "strategy": loaded_framework.strategy.value,
            "variants": {vid: v.to_dict() for vid, v in loaded_framework.variants.items()}
        }
        return jsonify(config)
    except Exception as e:
        # Return default structure if no config exists
        return jsonify({
            "experiment_name": "sentiment_v1",
            "strategy": "session_hash",
            "variants": {}
        })

@admin_bp.route('/api/experiment/save', methods=['POST'])
def save_experiment_config():
    data = request.json
    try:
        framework = ABTestingFramework(
            experiment_name=data.get('experiment_name', 'sentiment_v1'),
            strategy=TrafficSplitStrategy(data.get('strategy', 'session_hash'))
        )
        
        variants_data = data.get('variants', {})
        for vid, v_data in variants_data.items():
            variant = ModelVariant(
                variant_id=vid,
                model_path=v_data['model_path'],
                preprocessor_path=v_data['preprocessor_path'],
                weight=float(v_data.get('weight', 0.5)),
                description=v_data.get('description', '')
            )
            framework.add_variant(variant)
            
        framework.save_config()
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error saving experiment config: {e}")
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/api/models/finetune', methods=['POST'])
def run_finetune():
    try:
        new_version = finetune_model()
        if new_version:
            return jsonify({
                'success': True, 
                'message': f'Model fine-tuned successfully: {new_version.version}',
                'version': new_version.version
            })
        return jsonify({'success': False, 'message': 'No new feedback to process or fine-tuning failed.'})
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/api/models/rollback', methods=['POST'])
def run_rollback():
    try:
        registry = ModelRegistry()
        new_active = registry.rollback()
        if new_active:
            return jsonify({
                'success': True, 
                'message': f'Rolled back to version: {new_active.version}',
                'version': new_active.version
            })
        return jsonify({'success': False, 'message': 'No previous version found to rollback to.'})
    except Exception as e:
        logger.error(f"Error during rollback: {e}")
        return jsonify({'error': str(e)}), 500
