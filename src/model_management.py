import logging
import os
import shutil
from typing import Optional

from src.database import ModelVersion, get_db_session

logger = logging.getLogger(__name__)

# Get project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
MODELS_DIR = os.path.join(project_root, 'models')
MAIN_MODEL_PATH = os.path.join(MODELS_DIR, 'lstm_model.pth')

def activate_model(version_name: str) -> bool:
    """
    Sets a specific model version as the active one.
    Copies the versioned file to the main model path.
    """
    db_session = get_db_session()
    version = db_session.query(ModelVersion).filter(ModelVersion.version == version_name).first()

    if not version:
        db_session.close()
        raise ValueError(f"Version {version_name} not found in database.")

    if not os.path.exists(version.model_path):
        db_session.close()
        raise FileNotFoundError(f"Model file for version {version_name} not found at {version.model_path}")

    # Backup current model if it exists
    if os.path.exists(MAIN_MODEL_PATH):
        backup_path = MAIN_MODEL_PATH + ".bak"
        shutil.copy2(MAIN_MODEL_PATH, backup_path)
        logger.info(f"Backed up current model to {backup_path}")

    # Copy new version to main path
    shutil.copy2(version.model_path, MAIN_MODEL_PATH)
    logger.info(f"Activated model version {version_name}")

    # Update database: status='active' for new one, 'archived' for others
    db_session.query(ModelVersion).filter(ModelVersion.status == 'active').update({ModelVersion.status: 'archived'})
    version.status = 'active'
    db_session.commit()
    db_session.close()

    return True

def rollback_to_previous() -> bool:
    """
    Rolls back to the previously active version.
    """
    db_session = get_db_session()
    versions = db_session.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()

    if len(versions) < 2:
        db_session.close()
        raise ValueError("Not enough model versions to perform rollback.")

    # Find current active
    current_active = next((v for v in versions if v.status == 'active'), None)

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

def get_active_version() -> Optional[ModelVersion]:
    db_session = get_db_session()
    version = db_session.query(ModelVersion).filter(ModelVersion.status == 'active').first()
    db_session.close()
    return version

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    active = get_active_version()
    if active:
        print(f"Current Active Model: {active.version}")
        print(f"Path: {active.model_path}")
        print(f"Created: {active.created_at}")
    else:
        print("No active model found in the database.")
