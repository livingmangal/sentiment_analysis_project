import torch
import torch.nn as nn

class SentimentGRU(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 1, dropout: float = 0.0, bidirectional: bool = False):
        super(SentimentGRU, self).__init__()
        
        # Simple embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Single layer GRU
        self.lstm = nn.GRU(
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
        
        # Pass through GRU
        lstm_out, hidden = self.lstm(embedded)
        
        # Apply mask to outputs to ignore padding in average
        masked_out = lstm_out * mask
        
        # Global Average Pooling
        # Sum across sequence length and divide by actual length
        sum_out = torch.sum(masked_out, dim=1)
        actual_lengths = torch.sum(mask, dim=1).clamp(min=1)
        avg_pool = sum_out / actual_lengths
        
        dropped = self.dropout(avg_pool)
        output = self.fc(dropped)
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
    def load(cls, path: str) -> 'SentimentGRU':
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
