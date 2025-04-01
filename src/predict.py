import torch
from typing import Dict, Any
import os
from src.model import SentimentLSTM
from src.preprocessing import TextPreprocessor

class SentimentPredictor:
    def __init__(self, model_path: str, preprocessor_path: str):
        """Initialize the sentiment predictor with model and preprocessor"""
        # Ensure paths are absolute
        model_path = os.path.abspath(model_path)
        preprocessor_path = os.path.abspath(preprocessor_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
        
        # Load preprocessor
        self.preprocessor = TextPreprocessor.load(preprocessor_path)
        
        # Create model with same architecture as training
        self.model = SentimentLSTM(
            vocab_size=self.preprocessor.vocab_size,
            embedding_dim=32,
            hidden_dim=32,
            output_dim=1,
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
        """
        Predict sentiment for given text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing:
                - sentiment: "Positive" or "Negative"
                - confidence: float between 0 and 1
                - raw_score: raw model output
        """
        # Preprocess text
        text_tensor = self.preprocessor.transform(text)
        
        # Add batch dimension and move to device
        text_tensor = text_tensor.unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(text_tensor)
            probability = torch.sigmoid(output).item()
            
        # Determine sentiment and confidence
        sentiment = "Positive" if probability > 0.5 else "Negative"
        confidence = probability if sentiment == "Positive" else 1 - probability
        
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "raw_score": round(output.item(), 4)
        }

# Global predictor instance
_predictor = None

def initialize_predictor(model_path: str = 'models/lstm_model.pth',
                        preprocessor_path: str = 'models/preprocessor.pkl') -> None:
    """Initialize the global predictor instance"""
    global _predictor
    _predictor = SentimentPredictor(model_path, preprocessor_path)

def predict_sentiment(text: str) -> dict:
    """
    Predict sentiment for given text using the global predictor instance
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary containing sentiment prediction and confidence
    """
    global _predictor
    if _predictor is None:
        raise RuntimeError("Predictor not initialized. Call initialize_predictor() first.")
    return _predictor.predict(text)

if __name__ == "__main__":
    # Example usage
    try:
        initialize_predictor()
        test_text = "This movie was fantastic!"
        result = predict_sentiment(test_text)
        print(f'Text: {test_text}')
        print(f'Predicted Sentiment: {result["sentiment"]}')
        print(f'Confidence: {result["confidence"]}')
    except Exception as e:
        print(f"Error: {str(e)}")
