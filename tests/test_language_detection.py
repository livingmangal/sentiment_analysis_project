from src.language_detection import AdvancedLanguageDetector, LanguageDetector


class TestLanguageDetector:
    def test_detect_english(self):
        detector = LanguageDetector()
        assert detector.detect("This is an English sentence.") == "en"
        assert detector.detect("I love programming.") == "en"

    def test_detect_spanish(self):
        detector = LanguageDetector()
        assert detector.detect("el gato es azul") == "es" # 'el' is common
        assert detector.detect("la casa es grande") == "es" # 'la' is common

    def test_detect_french(self):
        detector = LanguageDetector()
        # French needs accents or specific words
        assert detector.detect("le chat est noir") == "fr" # 'le' is common
        assert detector.detect("je ne sais pas") == "fr" # 'je', 'ne' are common

    def test_detect_german(self):
        detector = LanguageDetector()
        assert detector.detect("das ist gut") == "de" # 'das' is common
        assert detector.detect("der mann ist hier") == "de" # 'der' is common

    def test_detect_hindi(self):
        detector = LanguageDetector()
        # Hindi
        assert detector.detect("यह एक हिंदी वाक्य है।") == "hi"

    def test_confidence(self):
        detector = LanguageDetector()
        conf = detector.get_confidence("This is English.")
        assert conf['en'] > conf['es']

class TestAdvancedLanguageDetector:
    def test_detect_with_fallback(self):
        # Even if langdetect is not installed, it should fallback to simple detector
        detector = AdvancedLanguageDetector()
        assert detector.detect("This is English.") == "en"
