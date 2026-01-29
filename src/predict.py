import torch
import sys
from typing import Dict, Any, List
import os
import json
import functools
import time
from src.model import SentimentLSTM
from src.preprocessing import TextPreprocessor

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

class SentimentPredictor:
    def __init__(self, model_path: str, preprocessor_path: str, metadata_path: str = None, quantize: bool = False): 
        """Initialize the sentiment predictor with model, preprocessor, and metadata"""
        # Ensure paths are absolute
        model_path = os.path.abspath(model_path)
        preprocessor_path = os.path.abspath(preprocessor_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")
        
        # Load Metadata
        self.metadata = {"version": "unknown", "training_date": "unknown"}
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")

        # Load preprocessor
        self.preprocessor = TextPreprocessor.load(preprocessor_path)
        
        # Load model using the class method
        self.model = SentimentLSTM.load(model_path)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Move model to CPU
        self.device = torch.device('cpu')
        self.model = self.model.to(self.device)

        # Model Quantization (Dynamic Quantization for CPU)
        if quantize:
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8
                )
                self.metadata["quantized"] = True
            except Exception as e:
                print(f"Warning: Model quantization failed: {e}")
                self.metadata["quantized"] = False
        else:
            self.metadata["quantized"] = False

    @functools.lru_cache(maxsize=1024)
    def predict(self, text: str) -> dict:
        """Predict sentiment for a single text with caching"""
        # Preprocess text
        text_tensor = self.preprocessor.transform(text)
        
        # Add batch dimension and move to device
        text_tensor = text_tensor.unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            start_time = time.time()
            output = self.model(text_tensor)
            inference_time = time.time() - start_time
            probability = torch.sigmoid(output).item()
            
        # Determine sentiment and confidence
        sentiment = "Positive" if probability > 0.5 else "Negative"
        confidence = probability if sentiment == "Positive" else 1 - probability
        
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "raw_score": round(output.item(), 4),
            "model_info": self.metadata,
            "inference_time_ms": round(inference_time * 1000, 2)
        }

    def predict_batch(self, texts: List[str]) -> List[dict]:
        """Predict sentiment for a batch of texts"""
        if not texts:
            return []
            
        # Preprocess all texts
        batch_tensor = self.preprocessor.transform_batch(texts).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(batch_tensor)
            inference_time = (time.time() - start_time) / len(texts)
            
            # Use view(-1) to safely handle batch sizes of 1 and above
            probabilities = torch.sigmoid(outputs).view(-1).tolist()
            raw_scores = outputs.view(-1).tolist()
            
        results = []
        for prob, raw in zip(probabilities, raw_scores):
            sentiment = "Positive" if prob > 0.5 else "Negative"
            confidence = prob if sentiment == "Positive" else 1 - prob
            results.append({
                "sentiment": sentiment,
                "confidence": round(confidence, 4),
                "raw_score": round(raw, 4),
                "model_info": self.metadata,
                "inference_time_ms": round(inference_time * 1000, 2)
            })
            
        return results

# Global predictor instance
_predictor = None

def initialize_predictor(model_path: str = None,
                         preprocessor_path: str = None,
                         metadata_path: str = None,
                         quantize: bool = True) -> None: 
    """Initialize the global predictor instance"""
    global _predictor
    
    # Set defaults if not provided
    if model_path is None:
        model_path = os.path.join(project_root, 'models', 'lstm_model.pth')
    if preprocessor_path is None:
        preprocessor_path = os.path.join(project_root, 'models', 'preprocessor.pkl')
    if metadata_path is None:
        metadata_path = os.path.join(project_root, 'models', 'metadata.json')
        
    _predictor = SentimentPredictor(model_path, preprocessor_path, metadata_path, quantize=quantize)

def predict_sentiment(text: str) -> dict:
    """Wrapper for global predictor single prediction"""
    global _predictor
    if _predictor is None:
        initialize_predictor()
    return _predictor.predict(text)

def predict_sentiment_batch(texts: List[str]) -> List[dict]:
    """Wrapper for global predictor batch prediction"""
    global _predictor
    if _predictor is None:
        initialize_predictor()
    return _predictor.predict_batch(texts)

if __name__ == "__main__":
    try:
        initialize_predictor()
        test_text = "This movie was fantastic!"
        result = predict_sentiment(test_text)
        print(json.dumps(result, indent=2)) # Pretty print to verify metadata
    except Exception as e:
        print(f"Error: {str(e)}")