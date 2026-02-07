import torch

from src.model import SentimentLSTM
from src.preprocessing import TextPreprocessor


class TestSentimentLSTM:
    def test_model_initialization(self):
        model = SentimentLSTM(vocab_size=100, embedding_dim=10, hidden_dim=20, output_dim=1)
        assert isinstance(model, SentimentLSTM)
        assert model.embedding.num_embeddings == 100
        assert model.lstm.hidden_size == 20

    def test_forward_pass(self):
        model = SentimentLSTM(vocab_size=100, embedding_dim=10, hidden_dim=20, output_dim=1)
        # Batch size 2, seq length 5
        x = torch.randint(0, 100, (2, 5))
        output = model(x)
        assert output.shape == (2, 1)

    def test_save_load(self, tmp_path):
        model = SentimentLSTM(vocab_size=100, embedding_dim=10, hidden_dim=20, output_dim=1)
        path = tmp_path / "model.pth"
        model.save(str(path))

        loaded_model = SentimentLSTM.load(str(path))
        assert loaded_model.embedding.num_embeddings == 100
        assert loaded_model.lstm.hidden_size == 20

class TestTextPreprocessor:
    def test_fit_transform(self):
        texts = ["hello world", "hello python"]
        preprocessor = TextPreprocessor(max_vocab_size=10, max_seq_length=5)
        preprocessor.fit(texts)

        assert preprocessor.vocab_size > 0
        assert "hello" in preprocessor.word_to_idx

        tensor = preprocessor.transform("hello world")
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (5,) # max_seq_length
        # Check padding (pad index is usually 0 if <pad> is first)
        # indices should be [pad, pad, pad, hello_idx, world_idx] or similar depending on implementation

        batch_tensor = preprocessor.transform_batch(texts)
        assert batch_tensor.shape == (2, 5)

    def test_save_load(self, tmp_path):
        texts = ["hello world"]
        preprocessor = TextPreprocessor(max_vocab_size=10, max_seq_length=5)
        preprocessor.fit(texts)

        path = tmp_path / "preprocessor.pkl"
        preprocessor.save(str(path))

        loaded = TextPreprocessor.load(str(path))
        assert loaded.vocab_size == preprocessor.vocab_size
        assert loaded.word_to_idx == preprocessor.word_to_idx
