from unittest.mock import MagicMock, patch

import pytest
import torch

from src.predict import SentimentPredictor


class TestSentimentPredictor:
    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.embedding.num_embeddings = 100
        model.eval.return_value = None
        model.to.return_value = model
        model.return_value = torch.tensor([[0.5]]) # Logits for single input (1, 1)
        return model

    @pytest.fixture
    def mock_preprocessor(self):
        prep = MagicMock()
        prep.transform.return_value = torch.zeros(10, dtype=torch.long)
        return prep

    @patch('src.predict.os.path.exists')
    @patch('src.predict.SentimentLSTM.load')
    @patch('src.predict.TextPreprocessor.load')
    def test_initialization(self, mock_prep_load, mock_model_load, mock_exists, mock_model, mock_preprocessor):
        mock_exists.return_value = True
        mock_model_load.return_value = mock_model
        mock_prep_load.return_value = mock_preprocessor

        predictor = SentimentPredictor("model.pth", "prep.pkl")
        assert predictor.model == mock_model
        assert predictor.preprocessor == mock_preprocessor

    @patch('src.predict.os.path.exists')
    @patch('src.predict.SentimentLSTM.load')
    @patch('src.predict.TextPreprocessor.load')
    def test_predict(self, mock_prep_load, mock_model_load, mock_exists, mock_model, mock_preprocessor):
        mock_exists.return_value = True
        mock_model_load.return_value = mock_model
        mock_prep_load.return_value = mock_preprocessor

        # Mock model output for predict
        # shape (1, 1) since we unsqueeze in predict
        mock_model.return_value = torch.tensor([[0.5]])

        predictor = SentimentPredictor("model.pth", "prep.pkl")
        result = predictor.predict("test text")

        assert "sentiment" in result
        assert "confidence" in result
        mock_preprocessor.transform.assert_called_with("test text")
        mock_model.assert_called()

    @patch('src.predict.os.path.exists')
    @patch('src.predict.SentimentLSTM.load')
    @patch('src.predict.TextPreprocessor.load')
    def test_predict_quantize_error(self, mock_prep_load, mock_model_load, mock_exists, mock_model, mock_preprocessor):
        mock_exists.return_value = True
        mock_model_load.return_value = mock_model
        mock_prep_load.return_value = mock_preprocessor

        # Test that quantization failure doesn't crash initialization
        with patch('torch.quantization.quantize_dynamic', side_effect=Exception("Quantization failed")):
            predictor = SentimentPredictor("model.pth", "prep.pkl", quantize=True)
            assert predictor.metadata["quantized"] is False

