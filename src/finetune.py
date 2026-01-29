import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import json
import logging
from datetime import datetime
from src.model import SentimentLSTM
from src.preprocessing import TextPreprocessor
from src.database import get_db_session, Feedback, ModelVersion
from src.registry import ModelRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def finetune_model(batch_size: int = 16, epochs: int = 5, lr: float = 0.0001):
    """
    Fine-tune the current active model using collected user feedback.
    """
    session = get_db_session()
    registry = ModelRegistry(session)
    
    # 1. Get latest active model
    active_model_version = registry.get_latest_active_model()
    if not active_model_version:
        logger.error("No active model found to fine-tune.")
        return None

    logger.info(f"Fine-tuning model version: {active_model_version.version}")
    
    # 2. Get unused feedback
    feedback_items = session.query(Feedback).filter_by(is_used_for_training=False).all()
    if not feedback_items:
        logger.info("No new feedback items to process.")
        return None

    logger.info(f"Found {len(feedback_items)} new feedback items.")
    
    # 3. Prepare data
    try:
        preprocessor = TextPreprocessor.load(active_model_version.preprocessor_path)
        model = SentimentLSTM.load(active_model_version.model_path)
    except Exception as e:
        logger.error(f"Error loading model or preprocessor: {e}")
        return None

    texts = [item.text for item in feedback_items]
    labels = [item.actual_sentiment for item in feedback_items]
    
    X = torch.stack([preprocessor.transform(text) for text in texts])
    y = torch.tensor(labels, dtype=torch.long)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
    
    # 4. Fine-tuning setup
    # Use a smaller learning rate for fine-tuning
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}")
    
    # 5. Save and register the new version
    version_str = f"ft_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_dir = os.path.join(project_root, 'models', 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_model_path = os.path.join(temp_dir, f'model_{version_str}.pth')
    temp_preproc_path = os.path.join(temp_dir, f'preprocessor_{version_str}.pkl')
    
    model.save(temp_model_path)
    preprocessor.save(temp_preproc_path)
    
    metrics = {
        "fine_tuning_samples": len(feedback_items),
        "base_version": active_model_version.version,
        "finetune_date": datetime.now().isoformat()
    }
    
    new_version = registry.register_model(
        model_path=temp_model_path,
        preprocessor_path=temp_preproc_path,
        metrics=metrics,
        version=version_str,
        status="staging" # New fine-tuned models go to staging
    )
    
    # 6. Mark feedback as used
    for item in feedback_items:
        item.is_used_for_training = True
    session.commit()
    
    # Cleanup temp files
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)
    if os.path.exists(temp_preproc_path):
        os.remove(temp_preproc_path)
        
    logger.info(f"Fine-tuning complete. New version: {new_version.version}")
    return new_version

if __name__ == "__main__":
    finetune_model()
