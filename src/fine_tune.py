import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import json
from datetime import datetime
from src.model import SentimentLSTM
from src.preprocessing import TextPreprocessor
from src.database import get_db_session, Feedback, ModelVersion

# Get project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
MODELS_DIR = os.path.join(project_root, 'models')

def fine_tune_model(batch_size=8, lr=0.0001, epochs=3):
    """
    Fine-tune the model using collected user feedback.
    """
    db_session = get_db_session()
    
    # 1. Get unused feedback
    unused_feedback = db_session.query(Feedback).filter(Feedback.is_used_for_training == False).all()
    
    if not unused_feedback:
        print("No new feedback to fine-tune on.")
        db_session.close()
        return None

    print(f"Fine-tuning on {len(unused_feedback)} new feedback samples...")

    # 2. Prepare data
    texts = [f.text for f in unused_feedback]
    labels = [f.correct_sentiment for f in unused_feedback]
    
    # 3. Load preprocessor and current model
    preprocessor = TextPreprocessor.load(os.path.join(MODELS_DIR, 'preprocessor.pkl'))
    model_path = os.path.join(MODELS_DIR, 'lstm_model.pth')
    
    if not os.path.exists(model_path):
        print(f"Base model not found at {model_path}. Cannot fine-tune.")
        db_session.close()
        return None
        
    model = SentimentLSTM.load(model_path)
    model.train()

    # 4. Transform data
    X = torch.stack([preprocessor.transform(t) for t in texts])
    y = torch.tensor(labels, dtype=torch.float32)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

    # 5. Fine-tune
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.view(-1), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.4f}")

    # 6. Save new model version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_version = f"ft_{timestamp}"
    new_model_path = os.path.join(MODELS_DIR, f"lstm_model_{new_version}.pth")
    model.save(new_model_path)
    
    # 7. Record new version in database
    model_version = ModelVersion(
        version=new_version,
        path=new_model_path,
        is_active=False
    )
    db_session.add(model_version)
    
    # 8. Mark feedback as used
    for f in unused_feedback:
        f.is_used_for_training = True
    
    db_session.commit()
    db_session.close()
    
    print(f"Fine-tuning complete. New model version: {new_version}")
    return new_version

if __name__ == "__main__":
    fine_tune_model()
