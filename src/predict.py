import torch
from typing import Dict, Any
import os
import json
from src.model import SentimentLSTM
from src.preprocessing import TextPreprocessor

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

class SentimentPredictor:
    def __init__(self, model_path: str, preprocessor_path: str, metadata_path: str = None): 
        """Initialize the sentiment predictor with model, preprocessor, and metadata"""
        # Ensure paths are absolute
        model_path = os.path.abspath(model_path)
        preprocessor_path = os.path.abspath(preprocessor_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
        
        # Load Metadata (New Section)
        self.metadata = {"version": "unknown", "training_date": "unknown"}
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")

        # Load preprocessor
        self.preprocessor = TextPreprocessor.load(preprocessor_path)
        
        # Create model with same architecture as training
        self.model = SentimentLSTM(
            vocab_size=self.preprocessor.vocab_size,
            embedding_dim=64,
            hidden_dim=64,
            output_dim=3,
            num_layers=1,
            dropout=0.2,
            bidirectional=True
        )
        
        # Load the trained weights
        self.model.load_state_dict(torch.load(model_path))
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Move model to CPU for Render deployment
        self.device = torch.device('cpu')
        self.model = self.model.to(self.device)
    
    def predict(self, text: str) -> dict:
        """Predict sentiment and include model metadata"""
        # Preprocess text
        text_tensor = self.preprocessor.transform(text)
        
        # Add batch dimension and move to device
        text_tensor = text_tensor.unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(text_tensor)
            probs = torch.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            
        # Determine sentiment and confidence
        sentiment_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
        sentiment = sentiment_map.get(pred_idx, "Unknown")
        confidence = probs[0][pred_idx].item()
        
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "raw_scores": [round(x, 4) for x in probs[0].tolist()],
            "model_info": self.metadata 
        }

# Global predictor instance
_predictor = None

def initialize_predictor(model_path: str = None,
                         preprocessor_path: str = None,
                         metadata_path: str = None) -> None: 
    """Initialize the global predictor instance"""
    global _predictor
    
    # Set defaults if not provided
    if model_path is None:
        model_path = os.path.join(project_root, 'models', 'lstm_model.pth')
    if preprocessor_path is None:
        preprocessor_path = os.path.join(project_root, 'models', 'preprocessor.pkl')
    if metadata_path is None:
        metadata_path = os.path.join(project_root, 'models', 'metadata.json')
        
    _predictor = SentimentPredictor(model_path, preprocessor_path, metadata_path)

def predict_sentiment(text: str) -> dict:
    """Wrapper for global predictor"""
    global _predictor
    if _predictor is None:
        raise RuntimeError("Predictor not initialized. Call initialize_predictor() first.")
    return _predictor.predict(text)

if __name__ == "__main__":
    try:
        initialize_predictor()
        test_text = "This movie was fantastic!"
        result = predict_sentiment(test_text)
        print(json.dumps(result, indent=2)) # Pretty print to verify metadata
    except Exception as e:
        print(f"Error: {str(e)}")