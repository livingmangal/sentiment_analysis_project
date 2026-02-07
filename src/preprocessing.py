import re
from collections import Counter
from typing import Dict, List, Set

import torch


class TextPreprocessor:
    def __init__(self, max_vocab_size: int = 10000, max_seq_length: int = 20):
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        self.stop_words: Set[str] = {
            'the', 'a', 'an', 'is', 'was', 'were', 'be', 'been', 'being',
            'to', 'of', 'and', 'or', 'but', 'in', 'on', 'at', 'with', 'for',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their',
            'this', 'that', 'these', 'those', 'am', 'are'
        }
        self.clean_re = re.compile(r'[^\w\s]')

    def fit(self, texts: List[str]) -> None:
        """Build vocabulary from list of texts"""
        # Clean and tokenize all texts
        words: List[str] = []
        for text in texts:
            words.extend(self._tokenize(text))

        # Create vocabulary
        word_counts = Counter(words)
        vocab = ['<pad>', '<unk>'] + [word for word, _ in word_counts.most_common(self.max_vocab_size - 2)]

        # Create word to index mapping
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(vocab)
        print(f"Vocab size: {self.vocab_size}")
        print(f"Common words: {vocab[2:12]}")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and extra whitespace using pre-compiled regex
        text = self.clean_re.sub(' ', text)
        # Split into words and filter stopwords
        return [word for word in text.split() if word and word not in self.stop_words]

    def transform(self, text: str) -> torch.Tensor:
        """Convert text to tensor"""
        if not self.word_to_idx:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        # Tokenize text
        words = self._tokenize(text)

        # Convert words to indices
        unk_idx = self.word_to_idx['<unk>']
        pad_idx = self.word_to_idx['<pad>']
        indices = [self.word_to_idx.get(word, unk_idx) for word in words]

        # Pad or truncate
        if len(indices) > self.max_seq_length:
            indices = indices[:self.max_seq_length]
        else:
            padding = [pad_idx] * (self.max_seq_length - len(indices))
            indices = padding + indices

        return torch.tensor(indices, dtype=torch.long)

    def transform_batch(self, texts: List[str]) -> torch.Tensor:
        """Convert list of texts to batch tensor"""
        if not self.word_to_idx:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        batch_indices = []
        unk_idx = self.word_to_idx['<unk>']
        pad_idx = self.word_to_idx['<pad>']

        for text in texts:
            words = self._tokenize(text)
            indices = [self.word_to_idx.get(word, unk_idx) for word in words]

            if len(indices) > self.max_seq_length:
                indices = indices[:self.max_seq_length]
            else:
                padding = [pad_idx] * (self.max_seq_length - len(indices))
                indices = padding + indices
            batch_indices.append(indices)

        return torch.tensor(batch_indices, dtype=torch.long)

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
