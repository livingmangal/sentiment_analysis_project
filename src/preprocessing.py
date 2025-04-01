import re
from collections import Counter
import torch
from typing import List, Tuple, Dict

class TextPreprocessor:
    def __init__(self, max_vocab_size: int = 10000, max_seq_length: int = 100):
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
    def fit(self, texts: List[str]) -> None:
        """Build vocabulary from list of texts"""
        # Clean and tokenize all texts
        words = []
        for text in texts:
            words.extend(self._tokenize(text))
        
        # Create vocabulary
        word_counts = Counter(words)
        vocab = ['<pad>', '<unk>'] + [word for word, _ in word_counts.most_common(self.max_vocab_size - 2)]
        
        # Create word to index mapping
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(vocab)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        # Split into words
        return [word for word in text.split() if word]
    
    def transform(self, text: str) -> torch.Tensor:
        """Convert text to tensor"""
        if not self.word_to_idx:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        # Tokenize text
        words = self._tokenize(text)
        
        # Convert words to indices
        indices = [self.word_to_idx.get(word, self.word_to_idx['<unk>']) for word in words]
        
        # Pad or truncate
        if len(indices) > self.max_seq_length:
            indices = indices[:self.max_seq_length]
        else:
            indices.extend([self.word_to_idx['<pad>']] * (self.max_seq_length - len(indices)))
        
        return torch.tensor(indices, dtype=torch.long)
    
    def save(self, path: str) -> None:
        """Save preprocessor state"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'word_to_idx': self.word_to_idx,
                'idx_to_word': self.idx_to_word,
                'vocab_size': self.vocab_size,
                'max_vocab_size': self.max_vocab_size,
                'max_seq_length': self.max_seq_length
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'TextPreprocessor':
        """Load preprocessor state"""
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        preprocessor = cls(state['max_vocab_size'], state['max_seq_length'])
        preprocessor.word_to_idx = state['word_to_idx']
        preprocessor.idx_to_word = state['idx_to_word']
        preprocessor.vocab_size = state['vocab_size']
        return preprocessor 