import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 1, dropout: float = 0.0, bidirectional: bool = False):
        super(SentimentLSTM, self).__init__()
        
        # Simple embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
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
        
        # Get embeddings
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_length, hidden_dim)
        
        # Get the last output
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Apply dropout
        dropped = self.dropout(last_hidden)
        
        # Pass through linear layer
        output = self.fc(dropped)  # (batch_size, output_dim)
        
        return output
    
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
        """Load model state"""
        checkpoint = torch.load(path)
        
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            embedding_dim=checkpoint['embedding_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            output_dim=checkpoint['output_dim'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout'],
            bidirectional=checkpoint['bidirectional']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
