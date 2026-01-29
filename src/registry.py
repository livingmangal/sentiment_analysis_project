from datetime import datetime
import json
import os
import shutil
from typing import List, Optional, Dict
from sqlalchemy.orm import Session
from src.database import ModelVersion, get_db_session

class ModelRegistry:
    def __init__(self, session: Session = None):
        self.session = session or get_db_session()

    def register_model(self, 
                       model_path: str, 
                       preprocessor_path: str, 
                       metrics: Dict[str, float], 
                       version: str = None,
                       status: str = "archived") -> ModelVersion:
        """
        Register a new model version.
        Moves the files to a versioned directory if they aren't already there.
        """
        if version is None:
            version = f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Define permanent storage path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        version_dir = os.path.join(project_root, 'models', 'versions', version)
        os.makedirs(version_dir, exist_ok=True)

        new_model_path = os.path.join(version_dir, 'model.pth')
        new_preproc_path = os.path.join(version_dir, 'preprocessor.pkl')

        # Copy files if they are not already in the target location
        if os.path.abspath(model_path) != os.path.abspath(new_model_path):
            shutil.copy2(model_path, new_model_path)
        
        if os.path.abspath(preprocessor_path) != os.path.abspath(new_preproc_path):
            shutil.copy2(preprocessor_path, new_preproc_path)

        # Create DB entry
        model_version = ModelVersion(
            version=version,
            model_path=new_model_path,
            preprocessor_path=new_preproc_path,
            metrics=json.dumps(metrics),
            status=status
        )
        
        self.session.add(model_version)
        self.session.commit()
        return model_version

    def get_model(self, version: str) -> Optional[ModelVersion]:
        return self.session.query(ModelVersion).filter_by(version=version).first()

    def list_models(self) -> List[ModelVersion]:
        return self.session.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()

    def set_status(self, version: str, status: str):
        model = self.get_model(version)
        if model:
            model.status = status
            self.session.commit()
            return True
        return False

    def get_latest_active_model(self) -> Optional[ModelVersion]:
        """Get the most recently created model with status 'active'"""
        return self.session.query(ModelVersion)\
            .filter_by(status='active')\
            .order_by(ModelVersion.created_at.desc())\
            .first()

    def rollback(self) -> Optional[ModelVersion]:
        """
        Rollback the current active model to the previous model (by ID).
        Returns the new active model version.
        """
        current_active = self.get_latest_active_model()
        if not current_active:
            # If no active model, try to activate the latest one
            latest = self.session.query(ModelVersion)\
                .order_by(ModelVersion.created_at.desc())\
                .first()
            if latest:
                latest.status = 'active'
                self.session.commit()
                return latest
            return None
        
        # Find the next most recent model
        previous_model = self.session.query(ModelVersion)\
            .filter(ModelVersion.id < current_active.id)\
            .order_by(ModelVersion.id.desc())\
            .first()
            
        if previous_model:
            current_active.status = 'archived'
            previous_model.status = 'active'
            self.session.commit()
            return previous_model
        
        return None
