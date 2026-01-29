"""
Database module for sentiment analysis application.
Provides SQLAlchemy models and database initialization.
"""
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import json

Base = declarative_base()


class Prediction(Base):
    """Model for storing prediction history"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    input_data = Column(Text, nullable=False)  # JSON string of input parameters
    prediction_result = Column(Text, nullable=False)  # JSON string of prediction output
    is_favorite = Column(Boolean, default=False, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    session_id = Column(String(255), nullable=False, index=True)
    
    def to_dict(self):
        """Convert prediction to dictionary"""
        return {
            'id': self.id,
            'input_data': json.loads(self.input_data) if isinstance(self.input_data, str) else self.input_data,
            'prediction_result': json.loads(self.prediction_result) if isinstance(self.prediction_result, str) else self.prediction_result,
            'is_favorite': self.is_favorite,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'session_id': self.session_id
        }


class ModelVersion(Base):
    """Model for storing trained model versions"""
    __tablename__ = 'model_versions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(50), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    model_path = Column(String(255), nullable=False)
    preprocessor_path = Column(String(255), nullable=False)
    metrics = Column(Text, nullable=True)  # JSON string of metrics
    status = Column(String(20), default="archived")  # active, staging, archived, deprecated

    def to_dict(self):
        return {
            'id': self.id,
            'version': self.version,
            'created_at': self.timestamp_iso(),
            'model_path': self.model_path,
            'preprocessor_path': self.preprocessor_path,
            'metrics': json.loads(self.metrics) if self.metrics else {},
            'status': self.status
        }
    
    def timestamp_iso(self):
        return self.created_at.isoformat() if self.created_at else None


# Database setup
def get_db_path():
    """Get the database file path"""
    # Get the project root directory (parent of src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    db_path = os.path.join(project_root, 'predictions.db')
    return db_path


def init_db():
    """Initialize the database and create tables"""
    db_path = get_db_path()
    # Use SQLite with check_same_thread=False for Flask compatibility
    engine = create_engine(f'sqlite:///{db_path}', connect_args={'check_same_thread': False})
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    return engine


def get_db_session(engine=None):
    """Get a database session"""
    if engine is None:
        engine = init_db()
    Session = sessionmaker(bind=engine)
    return Session()
