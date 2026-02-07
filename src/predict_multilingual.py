"""
Multilingual Sentiment Predictor
Uses XLM-RoBERTa for multilingual sentiment analysis
"""
import json
import os
import time
from typing import Any, Dict, List, Optional

import torch

from src.language_detection import AdvancedLanguageDetector
from src.model import SentimentLSTM
from src.preprocessing_multilingual import MultilingualTextPreprocessor
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import torch.nn.functional as F

from src.language_detection import AdvancedLanguageDetector


class MultilingualSentimentPredictor:
    """
    Sentiment predictor supporting multiple languages using a single XLM-RoBERTa model.
    """

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        auto_detect: bool = True,
        quantize: bool = False,
        cache_dir: str = None
    ):
        """
        Initialize multilingual predictor with XLM-RoBERTa
        
        Args:
            model_name: Name of the pre-trained model
            auto_detect: Whether to auto-detect language
            quantize: Whether to apply dynamic quantization
            cache_dir: Directory to cache model files
        """
        if models_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            models_dir = os.path.join(project_root, 'models', 'multilingual')

        self.models_dir = models_dir
        self.auto_detect = auto_detect
        self.quantize = quantize

        # Language detector
        self.language_detector = AdvancedLanguageDetector()

        # Storage for models and preprocessors
        self.models: Dict[str, SentimentLSTM] = {}
        self.preprocessors: Dict[str, MultilingualTextPreprocessor] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

        # Supported languages
        self.supported_languages = {'en', 'es', 'fr', 'de', 'hi'}

        # Device
        self.device = torch.device('cpu')

        # Load available models
        self._load_available_models()

    def _load_available_models(self) -> None:
        """Load all available language models"""
        if not os.path.exists(self.models_dir):
            print(f"Warning: Models directory not found: {self.models_dir}")
            print("Creating directory structure...")
            os.makedirs(self.models_dir, exist_ok=True)
            return

        for lang in self.supported_languages:
            model_path = os.path.join(self.models_dir, f'{lang}_model.pth')
            preprocessor_path = os.path.join(self.models_dir, f'{lang}_preprocessor.pkl')
            metadata_path = os.path.join(self.models_dir, f'{lang}_metadata.json')

            if os.path.exists(model_path) and os.path.exists(preprocessor_path):
                try:
                    self._load_language_model(lang, model_path, preprocessor_path, metadata_path)
                except Exception as e:
                    print(f"Error loading {lang} model: {e}")

    def _load_language_model(
        self,
        language: str,
        model_path: str,
        preprocessor_path: str,
        metadata_path: Optional[str] = None
    ) -> None:
        """Load model for specific language"""
        # Load preprocessor
        preprocessor = MultilingualTextPreprocessor.load(preprocessor_path)
        self.preprocessors[language] = preprocessor

        # Load model
        model = SentimentLSTM.load(model_path)
        model.eval()
        model = model.to(self.device)

        # Apply quantization if requested
        if self.quantize:
            try:
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear, torch.nn.LSTM}, dtype=torch.qint8
                )
            except Exception as e:
                print(f"Quantization failed for {language}: {e}")

        self.models[language] = model

        # Load metadata
        metadata = {"version": "1.0", "language": language}
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Could not load metadata for {language}: {e}")

        self.metadata[language] = metadata

        print(f"Loaded {language.upper()} model successfully")

    def add_language_model(
        self,
        language: str,
        model_path: str,
        preprocessor_path: str,
        metadata_path: Optional[str] = None
    ) -> None:
        """Add a new language model at runtime"""
        if language not in self.supported_languages:
            print(f"Warning: {language} is not in supported languages list")

        self._load_language_model(language, model_path, preprocessor_path, metadata_path)

    def predict(
        self,
        text: str,
        language: Optional[str] = None,
        return_language_confidence: bool = False
    ) -> Dict[str, Any]:
        """
        Predict sentiment for text using XLM-RoBERTa
        
        Args:
            text: Input text
            language: Language code (optional, used for metadata/tagging)
            return_language_confidence: Include language detection confidence
            
        Returns:
            Prediction result with sentiment, confidence, and metadata
        """
        # Detect language if not provided
        detected_language = language
        lang_confidence = None
        
        if language is None and self.auto_detect:
            detected_language = self.language_detector.detect(text)
            if return_language_confidence:
                lang_confidence = self.language_detector.get_confidence(text)
        elif language is None:
            language = 'en'  # Default to English

        # Check if model is available
        if language not in self.models:
            # Fallback to English if available
            if 'en' in self.models:
                print(f"Warning: No model for {language}, using English")
                language = 'en'
            else:
                raise ValueError(f"No model available for language: {language}")

        # Get model and preprocessor
        model = self.models[language]
        preprocessor = self.preprocessors[language]

        # Preprocess text
        text_tensor = preprocessor.transform(text, language)
        text_tensor = text_tensor.unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(**inputs)
            scores = outputs.logits[0].detach().cpu()
            probs = F.softmax(scores, dim=0).numpy()
            inference_time = time.time() - start_time
            
        # Map output to sentiment
        # cardiffnlp/twitter-xlm-roberta-base-sentiment outputs: 0 -> Negative, 1 -> Neutral, 2 -> Positive
        labels = ["Negative", "Neutral", "Positive"]
        max_idx = probs.argmax()
        sentiment = labels[max_idx]
        confidence = float(probs[max_idx])
        
        result = {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "raw_scores": [round(x, 4) for x in probs.tolist()],
            "language": detected_language,
            "model_info": {
                "name": self.model_name,
                "type": "XLM-RoBERTa Transformer",
                "quantized": self.quantize
            },
            "inference_time_ms": round(inference_time * 1000, 2)
        }
        
        if return_language_confidence and lang_confidence:
            result["language_detection"] = lang_confidence
            
        return result

    def predict_batch(
        self,
        texts: List[str],
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Predict sentiment for batch of texts
        
        Args:
            texts: List of input texts
            language: Language code (optional, used for metadata)
            
        Returns:
            List of prediction results
        """
        if not texts:
            return []
            
        # Detect languages if not provided (optional, mainly for metadata)
        detected_languages = [language] * len(texts)
        if language is None and self.auto_detect:
            # Simple detection loop (could be parallelized or batched if detector supports it)
            detected_languages = [self.language_detector.detect(t) for t in texts]

        # Tokenize batch
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        
        # Predict
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(**inputs)
            inference_time = (time.time() - start_time) / len(texts)
            
            scores = outputs.logits.detach().cpu()
            probs = F.softmax(scores, dim=1).numpy()
            
        results = []
        labels = ["Negative", "Neutral", "Positive"]
        
        for i, prob in enumerate(probs):
            max_idx = prob.argmax()
            results.append({
                "sentiment": labels[max_idx],
                "confidence": round(float(prob[max_idx]), 4),
                "raw_scores": [round(x, 4) for x in prob.tolist()],
                "language": detected_languages[i] if i < len(detected_languages) else "unknown",
                "model_info": {
                    "name": self.model_name,
                    "type": "XLM-RoBERTa Transformer",
                    "quantized": self.quantize
                },
                "inference_time_ms": round(inference_time * 1000, 2)
            })
            
        return results

    def get_available_languages(self) -> List[str]:
        """Get list of available languages (all supported by XLM-R)"""
        return list(self.supported_languages)
    
    def is_language_available(self, language: str) -> bool:
        """Check if language model is available"""
        # XLM-R supports many, but we can check specifically for requested ones
        return True # Generally true for XLM-R unless it's a very obscure language


# Global instance
_multilingual_predictor: Optional[MultilingualSentimentPredictor] = None


def initialize_multilingual_predictor(
    model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    auto_detect: bool = True,
    quantize: bool = True
) -> None:
    """Initialize global multilingual predictor"""
    global _multilingual_predictor
    _multilingual_predictor = MultilingualSentimentPredictor(
        model_name=model_name,
        auto_detect=auto_detect,
        quantize=quantize
    )


def predict_multilingual(
    text: str,
    language: Optional[str] = None
) -> Dict[str, Any]:
    """Predict sentiment with language detection"""
    global _multilingual_predictor

    if _multilingual_predictor is None:
        initialize_multilingual_predictor()

    if _multilingual_predictor is None:
         raise RuntimeError("Failed to initialize multilingual predictor")

    return _multilingual_predictor.predict(text, language)


def predict_multilingual_batch(
    texts: List[str],
    language: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Predict sentiment for batch"""
    global _multilingual_predictor

    if _multilingual_predictor is None:
        initialize_multilingual_predictor()

    if _multilingual_predictor is None:
         raise RuntimeError("Failed to initialize multilingual predictor")

    return _multilingual_predictor.predict_batch(texts, language)
