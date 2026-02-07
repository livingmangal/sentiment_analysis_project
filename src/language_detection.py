"""
Language Detection Module
Detects the language of input text to route to appropriate model
"""
import re
from typing import Dict


class LanguageDetector:
    """
    Simple rule-based language detector with character frequency analysis
    For production, consider using langdetect or FastText language identification
    """

    def __init__(self) -> None:
        # Character patterns for different languages
        self.patterns = {
            'en': {
                'chars': set('abcdefghijklmnopqrstuvwxyz'),
                'common_words': {'the', 'is', 'and', 'to', 'a', 'of', 'in', 'it', 'you', 'that'}
            },
            'es': {
                'chars': set('abcdefghijklmnopqrstuvwxyzáéíóúñü'),
                'common_words': {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se'}
            },
            'fr': {
                'chars': set('abcdefghijklmnopqrstuvwxyzàâäæçéèêëïîôùûüÿœ'),
                'common_words': {'le', 'de', 'un', 'être', 'et', 'à', 'il', 'avoir', 'ne', 'je'}
            },
            'de': {
                'chars': set('abcdefghijklmnopqrstuvwxyzäöüß'),
                'common_words': {'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich'}
            },
            'hi': {
                'chars': set('अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह'),
                'common_words': {'है', 'के', 'में', 'की', 'को', 'से', 'और', 'का', 'एक', 'पर'}
            }
        }

    def detect(self, text: str) -> str:
        """
        Detect language of input text
        
        Args:
            text: Input text
            
        Returns:
            Language code ('en', 'es', 'fr', 'de', 'hi')
        """
        if not text or len(text.strip()) == 0:
            return 'en'  # Default to English

        text_lower = text.lower()

        # Count character matches for each language
        scores: Dict[str, float] = {}

        for lang_code, patterns in self.patterns.items():
            score: float = 0.0

            # Character frequency score
            char_matches = sum(1 for c in text_lower if c in patterns['chars'])
            total_alpha = sum(1 for c in text_lower if c.isalpha())

            if total_alpha > 0:
                score += (float(char_matches) / float(total_alpha)) * 50.0

            # Common words score
            words = set(re.findall(r'\b\w+\b', text_lower))
            word_matches = len(words & patterns['common_words'])
            score += float(word_matches) * 10.0

            scores[lang_code] = score

        # Return language with highest score
        detected_lang = max(scores.items(), key=lambda x: x[1])[0]
        return str(detected_lang)

    def get_confidence(self, text: str) -> Dict[str, float]:
        """
        Get confidence scores for all languages
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping language codes to confidence scores (0-1)
        """
        if not text or len(text.strip()) == 0:
            return {lang: 0.0 for lang in self.patterns.keys()}

        text_lower = text.lower()
        scores: Dict[str, float] = {}

        for lang_code, patterns in self.patterns.items():
            score: float = 0.0

            # Character frequency
            char_matches = sum(1 for c in text_lower if c in patterns['chars'])
            total_alpha = sum(1 for c in text_lower if c.isalpha())

            if total_alpha > 0:
                score += (float(char_matches) / float(total_alpha)) * 50.0

            # Word matches
            words = set(re.findall(r'\b\w+\b', text_lower))
            word_matches = len(words & patterns['common_words'])
            score += float(word_matches) * 10.0

            scores[lang_code] = score

        # Normalize to 0-1 range
        max_score = float(max(scores.values()) if scores else 1.0)
        if max_score > 0:
            scores = {lang: float(score)/max_score for lang, score in scores.items()}

        return scores


# Integration with langdetect library (optional, for better accuracy)
class AdvancedLanguageDetector:
    """
    Advanced language detector using langdetect library
    Fallback to simple detector if library not available
    """

    def __init__(self) -> None:
        self.simple_detector = LanguageDetector()
        self.use_langdetect = False

        try:
            from langdetect import detect, detect_langs
            self.detect_func = detect
            self.detect_langs_func = detect_langs
            self.use_langdetect = True
        except ImportError:
            print("Warning: langdetect not installed. Using simple detector.")
            print("Install with: pip install langdetect")

    def detect(self, text: str) -> str:
        """Detect language with fallback"""
        if self.use_langdetect:
            try:
                lang = self.detect_func(text)
                # Map to supported languages
                supported = {'en', 'es', 'fr', 'de', 'hi'}
                return lang if lang in supported else 'en'
            except Exception:
                pass

        return self.simple_detector.detect(text)

    def get_confidence(self, text: str) -> Dict[str, float]:
        """Get confidence scores with fallback"""
        if self.use_langdetect:
            try:
                detections = self.detect_langs_func(text)
                scores = {d.lang: d.prob for d in detections}
                # Ensure all supported languages are included
                supported = {'en', 'es', 'fr', 'de', 'hi'}
                for lang in supported:
                    if lang not in scores:
                        scores[lang] = 0.0
                return scores
            except Exception:
                pass

        return self.simple_detector.get_confidence(text)
