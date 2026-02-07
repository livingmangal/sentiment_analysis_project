"""
Database module for sentiment analysis application.
Provides SQLAlchemy models and database initialization.
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import Boolean, DateTime, Integer, String, Text, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


# Define Base with type hinting support (SQLAlchemy 2.0+)
class Base(DeclarativeBase):
    pass

class Prediction(Base):
    """Model for storing prediction history"""
    __tablename__ = 'predictions'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    input_data: Mapped[str] = mapped_column(Text, nullable=False)  # JSON string
    prediction_result: Mapped[str] = mapped_column(Text, nullable=False)  # JSON string
    is_favorite: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    def to_dict(self) -> Dict[str, Any]:
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
    """Model for storing user feedback on predictions"""
    __tablename__ = 'feedback'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_sentiment: Mapped[Optional[int]] = mapped_column(Integer, nullable=True) # Index
    actual_sentiment: Mapped[int] = mapped_column(Integer, nullable=False)   # Index
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    is_used_for_training: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'prediction_id': self.prediction_id,
            'text': self.text,
            'predicted_sentiment': self.predicted_sentiment,
            'actual_sentiment': self.actual_sentiment,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'is_used_for_training': self.is_used_for_training
        }


class ModelVersion(Base):
    """Model for storing trained model versions"""
    __tablename__ = 'model_versions'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    model_path: Mapped[str] = mapped_column(String(255), nullable=False)
    preprocessor_path: Mapped[str] = mapped_column(String(255), nullable=False)
    metrics: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON string
    status: Mapped[str] = mapped_column(String(20), default="archived")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'version': self.version,
            'created_at': self.timestamp_iso(),
            'model_path': self.model_path,
            'preprocessor_path': self.preprocessor_path,
            'metrics': json.loads(self.metrics) if self.metrics and isinstance(self.metrics, str) else {},
            'status': self.status
        }

    def timestamp_iso(self) -> Optional[str]:
        return self.created_at.isoformat() if self.created_at else None


# Database setup
def get_db_path() -> str:
    """Get the database file path"""
    # Get the project root directory (parent of src/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    db_path = os.path.join(project_root, 'predictions.db')
    return db_path


def init_db() -> Engine:
    """Initialize the database and create tables"""
    db_path = get_db_path()
    # Use SQLite with check_same_thread=False for Flask compatibility
    engine = create_engine(f'sqlite:///{db_path}', connect_args={'check_same_thread': False})

    # Create all tables
    Base.metadata.create_all(engine)

    return engine


def get_db_session(engine: Optional[Engine] = None) -> Session:
    """Get a database session"""
    if engine is None:
        engine = init_db()

    # Use generic Session type to avoid strict mypy errors with sessionmaker return type
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()
