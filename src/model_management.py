import os
import shutil
from src.database import get_db_session, ModelVersion
import logging

logger = logging.getLogger(__name__)

# Get project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
MODELS_DIR = os.path.join(project_root, 'models')
MAIN_MODEL_PATH = os.path.join(MODELS_DIR, 'lstm_model.pth')

def activate_model(version_name: str):
    """
    Sets a specific model version as the active one.
    Copies the versioned file to the main model path.
    """
    db_session = get_db_session()
    version = db_session.query(ModelVersion).filter(ModelVersion.version == version_name).first()
    
    if not version:
        db_session.close()
        raise ValueError(f"Version {version_name} not found in database.")
    
    if not os.path.exists(version.path):
        db_session.close()
        raise FileNotFoundError(f"Model file for version {version_name} not found at {version.path}")

    # Backup current model if it exists
    if os.path.exists(MAIN_MODEL_PATH):
        backup_path = MAIN_MODEL_PATH + ".bak"
        shutil.copy2(MAIN_MODEL_PATH, backup_path)
        logger.info(f"Backed up current model to {backup_path}")

    # Copy new version to main path
    shutil.copy2(version.path, MAIN_MODEL_PATH)
    logger.info(f"Activated model version {version_name}")

    # Update database
    db_session.query(ModelVersion).update({ModelVersion.is_active: False})
    version.is_active = True
    db_session.commit()
    db_session.close()
    
    return True

def rollback_to_previous():
    """
    Rolls back to the previously active version.
    """
    db_session = get_db_session()
    versions = db_session.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
    
    if len(versions) < 2:
        db_session.close()
        raise ValueError("Not enough model versions to perform rollback.")

    # Find current active
    current_active = next((v for v in versions if v.is_active), None)
    
    # Find next version to activate
    target_version = None
    if current_active:
        target_version = db_session.query(ModelVersion).filter(
            ModelVersion.created_at < current_active.created_at
        ).order_by(ModelVersion.created_at.desc()).first()
    
    if not target_version:
        target_version = versions[1]

    db_session.close()
    return activate_model(target_version.version)

def get_active_version():
    db_session = get_db_session()
    version = db_session.query(ModelVersion).filter(ModelVersion.is_active == True).first()
    db_session.close()
    return version
