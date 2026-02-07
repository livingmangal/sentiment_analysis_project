from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch
import torch.nn.functional as F
from typing import Dict, List, Any
import os

class SentimentTransformer:
    def __init__(self, model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment", cache_dir: str = None):
        """
        Initialize the transformer model for sentiment analysis.
        Args:
            model_name: The name of the pre-trained model to use.
            cache_dir: Directory to cache the model files.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        print(f"Loading transformer model: {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
            self.config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a single text.
        Returns dictionary with sentiment, confidence, and raw scores.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits[0].detach().cpu()
            probs = F.softmax(scores, dim=0).numpy()

        # Mapping specific to cardiffnlp/twitter-xlm-roberta-base-sentiment
        # 0: Negative, 1: Neutral, 2: Positive
        labels = ["Negative", "Neutral", "Positive"]
        
        # Get the label with the highest probability
        max_idx = probs.argmax()
        sentiment = labels[max_idx]
        confidence = float(probs[max_idx])
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "raw_scores": probs.tolist()
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for a batch of texts.
        """
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.detach().cpu()
            probs = F.softmax(scores, dim=1).numpy()
            
        results = []
        labels = ["Negative", "Neutral", "Positive"]
        
        for prob in probs:
            max_idx = prob.argmax()
            results.append({
                "sentiment": labels[max_idx],
                "confidence": float(prob[max_idx]),
                "raw_scores": prob.tolist()
            })
            
        return results
