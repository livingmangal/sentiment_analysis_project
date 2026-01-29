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


class Feedback(Base):
    """Model for storing user feedback for fine-tuning"""
    __tablename__ = 'feedback'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, nullable=True) # Link to prediction (optional)
    text = Column(Text, nullable=False) # Original text
    predicted_sentiment = Column(String(20), nullable=True) # Sentiment model predicted
    correct_sentiment = Column(Integer, nullable=False) # 0 for Negative, 1 for Positive
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_used_for_training = Column(Boolean, default=False, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'prediction_id': self.prediction_id,
            'text': self.text,
            'predicted_sentiment': self.predicted_sentiment,
            'correct_sentiment': self.correct_sentiment,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'is_used_for_training': self.is_used_for_training
        }


class ModelVersion(Base):
    """Model for tracking model versions and rollback"""
    __tablename__ = 'model_versions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(50), nullable=False, unique=True)
    path = Column(String(255), nullable=False)
    accuracy = Column(String(20), nullable=True)
    is_active = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'version': self.version,
            'path': self.path,
            'accuracy': self.accuracy,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


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
