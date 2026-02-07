"""
Multilingual Sentiment Predictor
Extends SentimentPredictor to support multiple languages
"""
import json
import os
import time
from typing import Any, Dict, List, Optional

import torch

from src.language_detection import AdvancedLanguageDetector
from src.model import SentimentLSTM
from src.preprocessing_multilingual import MultilingualTextPreprocessor


class MultilingualSentimentPredictor:
    """
    Sentiment predictor supporting multiple languages
    Each language has its own model and preprocessor
    """

    def __init__(
        self,
        models_dir: Optional[str] = None,
        auto_detect: bool = True,
        quantize: bool = False
    ):
        """
        Initialize multilingual predictor
        
        Args:
            models_dir: Directory containing language-specific models
            auto_detect: Whether to auto-detect language
            quantize: Whether to apply dynamic quantization
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
        Predict sentiment for text
        
        Args:
            text: Input text
            language: Language code (auto-detected if None)
            return_language_confidence: Include language detection confidence
            
        Returns:
            Prediction result with sentiment, confidence, and metadata
        """
        # Detect language if not provided
        if language is None and self.auto_detect:
            language = self.language_detector.detect(text)
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
            output = model(text_tensor)
            inference_time = time.time() - start_time

            probability = torch.sigmoid(output).item()

        # Determine sentiment
        sentiment = "Positive" if probability > 0.5 else "Negative"
        confidence = probability if sentiment == "Positive" else 1 - probability

        result = {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "raw_score": round(output.item(), 4),
            "language": language,
            "model_info": self.metadata.get(language, {}),
            "inference_time_ms": round(inference_time * 1000, 2)
        }

        # Add language detection confidence if requested
        if return_language_confidence:
            lang_confidence = self.language_detector.get_confidence(text)
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
            language: Language code (auto-detected per text if None)
            
        Returns:
            List of prediction results
        """
        if not texts:
            return []

        # If language specified, process all together
        if language is not None:
            if language not in self.models:
                raise ValueError(f"No model for language: {language}")

            model = self.models[language]
            preprocessor = self.preprocessors[language]

            # Preprocess batch
            batch_tensor = preprocessor.transform_batch(texts, language).to(self.device)

            # Predict
            with torch.no_grad():
                start_time = time.time()
                outputs = model(batch_tensor)
                inference_time = (time.time() - start_time) / len(texts)

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
                    "language": language,
                    "model_info": self.metadata.get(language, {}),
                    "inference_time_ms": round(inference_time * 1000, 2)
                })

            return results

        # Otherwise, detect language for each text and process individually
        results = []
        for text in texts:
            result = self.predict(text, language=None)
            results.append(result)

        return results

    def get_available_languages(self) -> List[str]:
        """Get list of available languages"""
        return list(self.models.keys())

    def is_language_available(self, language: str) -> bool:
        """Check if language model is available"""
        return language in self.models


# Global instance
_multilingual_predictor: Optional[MultilingualSentimentPredictor] = None


def initialize_multilingual_predictor(
    models_dir: Optional[str] = None,
    auto_detect: bool = True,
    quantize: bool = True
) -> None:
    """Initialize global multilingual predictor"""
    global _multilingual_predictor
    _multilingual_predictor = MultilingualSentimentPredictor(
        models_dir=models_dir,
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
