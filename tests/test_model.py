from src.model import SentimentLSTM
import torch

def test_model_output():
    model = SentimentLSTM(5000, 50, 128, 1)
    text_tensor = torch.randint(0, 5000, (1, 100))
    output = model(text_tensor)
    assert output.shape == (1, 1), "Model output shape is incorrect!"

test_model_output()
