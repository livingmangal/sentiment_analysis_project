from unittest.mock import MagicMock, patch

import pytest
import torch

from src.predict_multilingual import (
    MultilingualSentimentPredictor,
    initialize_multilingual_predictor,
    predict_multilingual,
)


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.eval.return_value = None
    model.to.return_value = model
    model.return_value = torch.tensor([[0.5]]) # Logits shape (1, 1)
    return model

@pytest.fixture
def mock_preprocessor():
    prep = MagicMock()
    prep.transform.return_value = torch.zeros(10, dtype=torch.long)
    prep.transform_batch.return_value = torch.zeros(2, 10, dtype=torch.long)
    return prep

class TestMultilingualPredictor:

    @patch('src.predict_multilingual.AdvancedLanguageDetector')
    @patch('src.predict_multilingual.os.path.exists')
    @patch('src.predict_multilingual.SentimentLSTM.load')
    @patch('src.predict_multilingual.MultilingualTextPreprocessor.load')
    def test_initialization_and_predict(self, mock_prep_load, mock_model_load, mock_exists, mock_lang_detector, mock_model, mock_preprocessor):
        # Setup mocks
        mock_exists.return_value = True # All paths exist
        mock_model_load.return_value = mock_model
        mock_prep_load.return_value = mock_preprocessor

        detector_instance = mock_lang_detector.return_value
        detector_instance.detect.return_value = 'en'

        predictor = MultilingualSentimentPredictor(models_dir="models")

        # Verify models loaded for all supported langs
        assert 'en' in predictor.models
        assert 'es' in predictor.models

        # Test predict with explicit language
        result = predictor.predict("Hello", language="en")
        assert result['sentiment'] == "Positive"
        assert result['language'] == "en"

        # Test predict with auto-detect
        result_auto = predictor.predict("Hello")
        assert result_auto['language'] == "en"
        detector_instance.detect.assert_called()

    @patch('src.predict_multilingual.AdvancedLanguageDetector')
    @patch('src.predict_multilingual.os.path.exists')
    @patch('src.predict_multilingual.SentimentLSTM.load')
    @patch('src.predict_multilingual.MultilingualTextPreprocessor.load')
    def test_predict_batch(self, mock_prep_load, mock_model_load, mock_exists, mock_lang_detector, mock_model, mock_preprocessor):
        mock_exists.return_value = True
        mock_model_load.return_value = mock_model
        mock_prep_load.return_value = mock_preprocessor

        # Mock batch input and output
        mock_model.return_value = torch.tensor([[0.1], [0.8]]) # Shape (2, 1)

        predictor = MultilingualSentimentPredictor(models_dir="models")

        # Batch predict with language
        results = predictor.predict_batch(["Hello", "World"], language="en")
        assert len(results) == 2

        # Batch predict without language (loops)
        # Reset model return value because predict() calls model individually (unsqueeze)
        mock_model.return_value = torch.tensor([[0.5]])
        results_auto = predictor.predict_batch(["Hello", "World"])
        assert len(results_auto) == 2

    @patch('src.predict_multilingual.AdvancedLanguageDetector')
    @patch('src.predict_multilingual.os.path.exists')
    @patch('src.predict_multilingual.SentimentLSTM.load')
    @patch('src.predict_multilingual.MultilingualTextPreprocessor.load')
    def test_wrappers(self, mock_prep_load, mock_model_load, mock_exists, mock_lang_detector, mock_model, mock_preprocessor):
        mock_exists.return_value = True
        mock_model_load.return_value = mock_model
        mock_prep_load.return_value = mock_preprocessor
        mock_model.return_value = torch.tensor([[0.5]])

        # Reset global
        import src.predict_multilingual
        src.predict_multilingual._multilingual_predictor = None

        # Initialize
        initialize_multilingual_predictor(models_dir="models")
        assert src.predict_multilingual._multilingual_predictor is not None

        # Predict wrapper
        res = predict_multilingual("Hello")
        assert res['sentiment'] == "Positive"
