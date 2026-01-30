"""
Multilingual Text Preprocessing
Extends TextPreprocessor to support multiple languages
"""
import re
from collections import Counter
import torch
from typing import List, Dict, Optional
import pickle


class MultilingualTextPreprocessor:
    """
    Text preprocessor supporting multiple languages
    Each language has its own vocabulary and stopwords
    """
    
    def __init__(self, max_vocab_size: int = 10000, max_seq_length: int = 20):
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        
        # Language-specific vocabularies
        self.language_vocabs: Dict[str, Dict] = {}
        
        # Language-specific stopwords
        self.stopwords = {
            'en': {
                'the', 'a', 'an', 'is', 'was', 'were', 'be', 'been', 'being',
                'to', 'of', 'and', 'or', 'but', 'in', 'on', 'at', 'with', 'for',
                'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
                'she', 'her', 'it', 'its', 'they', 'them', 'their', 'this', 'that'
            },
            'es': {
                'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'de', 'del',
                'a', 'al', 'en', 'con', 'por', 'para', 'y', 'o', 'pero', 'que',
                'es', 'son', 'está', 'están', 'ser', 'estar', 'yo', 'tú', 'él',
                'ella', 'nosotros', 'vosotros', 'ellos', 'ellas', 'mi', 'tu', 'su'
            },
            'fr': {
                'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'à', 'au',
                'aux', 'en', 'avec', 'pour', 'par', 'et', 'ou', 'mais', 'que',
                'est', 'sont', 'être', 'avoir', 'je', 'tu', 'il', 'elle', 'nous',
                'vous', 'ils', 'elles', 'mon', 'ton', 'son', 'ce', 'cette', 'ces'
            },
            'de': {
                'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einen',
                'einem', 'einer', 'und', 'oder', 'aber', 'in', 'an', 'auf', 'mit',
                'von', 'zu', 'bei', 'für', 'ist', 'sind', 'war', 'waren', 'sein',
                'haben', 'ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr', 'mein', 'dein'
            },
            'hi': {
                'है', 'हैं', 'था', 'थे', 'हो', 'होना', 'के', 'की', 'को', 'से',
                'में', 'पर', 'और', 'या', 'लेकिन', 'कि', 'एक', 'यह', 'वह', 'ये',
                'वे', 'मैं', 'तुम', 'आप', 'हम', 'मेरा', 'तुम्हारा', 'उसका', 'इसका'
            }
        }
        
        # Character normalization patterns for each language
        self.clean_patterns = {
            'en': re.compile(r'[^\w\s]'),
            'es': re.compile(r'[^\wáéíóúñü\s]', re.IGNORECASE),
            'fr': re.compile(r'[^\wàâäæçéèêëïîôùûüÿœ\s]', re.IGNORECASE),
            'de': re.compile(r'[^\wäöüß\s]', re.IGNORECASE),
            'hi': re.compile(r'[^\u0900-\u097F\s]')  # Devanagari script
        }
    
    def fit(self, texts: List[str], language: str = 'en') -> None:
        """
        Build vocabulary for specific language
        
        Args:
            texts: List of training texts
            language: Language code ('en', 'es', 'fr', 'de', 'hi')
        """
        if language not in self.stopwords:
            raise ValueError(f"Unsupported language: {language}")
        
        # Clean and tokenize all texts
        words = []
        for text in texts:
            words.extend(self._tokenize(text, language))
        
        # Create vocabulary
        word_counts = Counter(words)
        vocab = ['<pad>', '<unk>'] + [
            word for word, _ in word_counts.most_common(self.max_vocab_size - 2)
        ]
        
        # Store language-specific vocabulary
        self.language_vocabs[language] = {
            'word_to_idx': {word: idx for idx, word in enumerate(vocab)},
            'idx_to_word': {idx: word for idx, word in enumerate(vocab)},
            'vocab_size': len(vocab)
        }
        
        print(f"[{language.upper()}] Vocab size: {len(vocab)}")
        print(f"[{language.upper()}] Sample words: {vocab[2:12]}")
    
    def _tokenize(self, text: str, language: str) -> List[str]:
        """
        Tokenize text for specific language
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Get language-specific pattern
        pattern = self.clean_patterns.get(language, self.clean_patterns['en'])
        
        # Remove special characters
        text = pattern.sub(' ', text)
        
        # Get language-specific stopwords
        stopwords = self.stopwords.get(language, set())
        
        # Split and filter
        return [
            word for word in text.split()
            if word and word not in stopwords
        ]
    
    def transform(self, text: str, language: str = 'en') -> torch.Tensor:
        """
        Convert text to tensor for specific language
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            Tensor of word indices
        """
        if language not in self.language_vocabs:
            raise ValueError(f"No vocabulary for language: {language}. Call fit() first.")
        
        vocab = self.language_vocabs[language]
        
        # Tokenize text
        words = self._tokenize(text, language)
        
        # Convert to indices
        unk_idx = vocab['word_to_idx']['<unk>']
        pad_idx = vocab['word_to_idx']['<pad>']
        indices = [vocab['word_to_idx'].get(word, unk_idx) for word in words]
        
        # Pad or truncate
        if len(indices) > self.max_seq_length:
            indices = indices[:self.max_seq_length]
        else:
            padding = [pad_idx] * (self.max_seq_length - len(indices))
            indices = padding + indices
        
        return torch.tensor(indices, dtype=torch.long)
    
    def transform_batch(self, texts: List[str], language: str = 'en') -> torch.Tensor:
        """
        Convert batch of texts to tensor
        
        Args:
            texts: List of input texts
            language: Language code
            
        Returns:
            Batch tensor
        """
        return torch.stack([self.transform(text, language) for text in texts])
    
    def get_vocab_size(self, language: str = 'en') -> int:
        """Get vocabulary size for specific language"""
        if language not in self.language_vocabs:
            return 0
        return self.language_vocabs[language]['vocab_size']
    
    def save(self, path: str) -> None:
        """Save preprocessor state"""
        state = {
            'max_vocab_size': self.max_vocab_size,
            'max_seq_length': self.max_seq_length,
            'language_vocabs': self.language_vocabs,
            'stopwords': self.stopwords
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: str) -> 'MultilingualTextPreprocessor':
        """Load preprocessor state"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        preprocessor = cls(
            state['max_vocab_size'],
            state['max_seq_length']
        )
        preprocessor.language_vocabs = state['language_vocabs']
        
        # Update stopwords if saved (for backward compatibility)
        if 'stopwords' in state:
            preprocessor.stopwords.update(state['stopwords'])
        
        return preprocessor