from unittest.mock import MagicMock, patch

import pytest
import torch

from src.predict import SentimentPredictor, predict_sentiment, predict_sentiment_batch


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.embedding.num_embeddings = 100
    model.eval.return_value = None
    model.to.return_value = model
    model.return_value = torch.tensor([[0.1, 0.9]]) # Logits for single input (Positive)
    return model

@pytest.fixture
def mock_preprocessor():
    prep = MagicMock()
    prep.transform.return_value = torch.zeros(10, dtype=torch.long)
    prep.transform_batch.return_value = torch.zeros(2, 10, dtype=torch.long)
    return prep

class TestPredictCoverage:

    @patch('src.predict.os.path.exists')
    def test_init_missing_model(self, mock_exists):
        # First exist check (model) fails
        mock_exists.side_effect = [False]
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            SentimentPredictor("missing.pth", "prep.pkl")

    @patch('src.predict.os.path.exists')
    def test_init_missing_preprocessor(self, mock_exists):
        # First exist (model) -> True, Second (prep) -> False
        mock_exists.side_effect = [True, False]
        with pytest.raises(FileNotFoundError, match="Preprocessor file not found"):
            SentimentPredictor("model.pth", "missing.pkl")

    @patch('src.predict.os.path.exists')
    @patch('src.predict.SentimentLSTM.load')
    @patch('src.predict.TextPreprocessor.load')
    def test_predict_batch(self, mock_prep_load, mock_model_load, mock_exists, mock_model, mock_preprocessor):
        mock_exists.return_value = True
        mock_model_load.return_value = mock_model
        mock_prep_load.return_value = mock_preprocessor

        # Mock batch output
        mock_model.return_value = torch.tensor([[0.1, 0.9], [0.9, 0.1]])

        predictor = SentimentPredictor("model.pth", "prep.pkl")
        results = predictor.predict_batch(["text1", "text2"])

        assert len(results) == 2
        assert results[0]['sentiment'] == "Positive"
        assert results[1]['sentiment'] == "Negative"
        mock_preprocessor.transform_batch.assert_called_with(["text1", "text2"])

    @patch('src.predict.os.path.exists')
    @patch('src.predict.SentimentLSTM.load')
    @patch('src.predict.TextPreprocessor.load')
    def test_wrappers(self, mock_prep_load, mock_model_load, mock_exists, mock_model, mock_preprocessor):
        mock_exists.return_value = True
        mock_model_load.return_value = mock_model
        mock_prep_load.return_value = mock_preprocessor
        mock_model.return_value = torch.tensor([[0.1, 0.9]])

        # Reset global predictor
        import src.predict
        src.predict._predictor = None

        # Test predict_sentiment
        res = predict_sentiment("test")
        assert res['sentiment'] == "Positive"

        # Test predict_sentiment_batch
        src.predict._predictor = None # Reset again just in case
        mock_model.return_value = torch.tensor([[0.1, 0.9]])
        res_batch = predict_sentiment_batch(["test"])
        assert len(res_batch) == 1

    @patch('src.predict.os.path.exists')
    @patch('src.predict.SentimentLSTM.load')
    @patch('src.predict.TextPreprocessor.load')
    def test_quantization_success(self, mock_prep_load, mock_model_load, mock_exists, mock_model, mock_preprocessor):
        mock_exists.return_value = True
        mock_model_load.return_value = mock_model
        mock_prep_load.return_value = mock_preprocessor

        # Mock quantization
        with patch('torch.quantization.quantize_dynamic') as mock_quant:
            mock_quant.return_value = mock_model
            predictor = SentimentPredictor("model.pth", "prep.pkl", quantize=True)
            assert predictor.metadata["quantized"] is True
