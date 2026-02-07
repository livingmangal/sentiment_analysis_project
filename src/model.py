import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 1, dropout: float = 0.0, bidirectional: bool = False):
        super(SentimentLSTM, self).__init__()

        # Simple embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Single layer LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, output_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_length)
        # Create mask for padding (assuming 0 is padding)
        mask = (x != 0).float().unsqueeze(-1)

        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)

        # Pass through LSTM
        # LSTM returns (output, (hidden, cell))
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Apply mask to outputs to ignore padding in average
        masked_out = lstm_out * mask

        # Global Average Pooling
        # Sum across sequence length and divide by actual length
        sum_out = torch.sum(masked_out, dim=1)
        actual_lengths = torch.sum(mask, dim=1).clamp(min=1)
        avg_pool = sum_out / actual_lengths

        dropped = self.dropout(avg_pool)
        output = self.fc(dropped)
        return output if isinstance(output, torch.Tensor) else torch.tensor(output)

    def save(self, path: str) -> None:
        """Save model state"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.embedding.num_embeddings,
            'embedding_dim': self.embedding.embedding_dim,
            'hidden_dim': self.lstm.hidden_size,
            'output_dim': self.fc.out_features,
            'num_layers': self.lstm.num_layers,
            'dropout': self.dropout.p,
            'bidirectional': self.lstm.bidirectional
        }, path)

    @classmethod
    def load(cls, path: str) -> 'SentimentLSTM':
        """Load model state from path"""
        # Ensure model is loaded to CPU to avoid device mismatch issues
        device = torch.device('cpu')
        checkpoint = torch.load(path, map_location=device)

        # Extract model parameters from checkpoint
        model = cls(
            vocab_size=checkpoint.get('vocab_size', 10000),
            embedding_dim=checkpoint.get('embedding_dim', 100),
            hidden_dim=checkpoint.get('hidden_dim', 128),
            output_dim=checkpoint.get('output_dim', 1),
            num_layers=checkpoint.get('num_layers', 1),
            dropout=checkpoint.get('dropout', 0.0),
            bidirectional=checkpoint.get('bidirectional', False)
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval() # Set to evaluation mode by default
        return model
