import pytest
import json
from unittest.mock import patch, MagicMock
from app.api import app
from app.api_multilingual import multilingual_bp

@pytest.fixture
def client():
    """Sets up a fake client to test the app without running the server"""
    app.config['TESTING'] = True
    # We need to register the blueprint here if not already done in app/api.py
    # But checking Step 61, it IS registered.
    
    with app.test_client() as client:
        yield client

@pytest.fixture(autouse=True)
def mock_predictor():
    """Mock the multilingual predictor to avoid loading large models"""
    with patch('app.api_multilingual.predict_multilingual') as mock_predict, \
         patch('app.api_multilingual.predict_multilingual_batch') as mock_batch, \
         patch('app.api_multilingual.initialize_multilingual_predictor') as mock_init, \
         patch('src.predict_multilingual._multilingual_predictor') as mock_obj:
        
        # Setup mock return values
        mock_predict.return_value = {
            "sentiment": "Positive",
            "confidence": 0.95,
            "raw_score": 0.95,
            "language": "en",
            "model_info": {"name": "mock-model"},
            "inference_time_ms": 10.0
        }
        
        mock_batch.return_value = [
            {
                "sentiment": "Positive",
                "confidence": 0.95,
                "raw_score": 0.95,
                "language": "en",
                "model_info": {"name": "mock-model"},
                "inference_time_ms": 10.0
            },
            {
                "sentiment": "Negative",
                "confidence": 0.85,
                "raw_score": 0.15,
                "language": "en",
                "model_info": {"name": "mock-model"},
                "inference_time_ms": 10.0
            }
        ]
        
        # Mock language detector for the specific test that uses it directly via the global object
        mock_detector = MagicMock()
        mock_detector.detect.return_value = 'en'
        mock_detector.get_confidence.return_value = {'en': 0.99}
        mock_obj.language_detector = mock_detector
        
        yield {
            'predict': mock_predict,
            'batch': mock_batch,
            'init': mock_init,
            'obj': mock_obj
        }


@pytest.fixture
def client():
    """Sets up a fake client to test the app without running the server"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestMultilingualPredict:
    """Tests for /predict/multilingual endpoint"""
    
    def test_predict_multilingual_english(self, client):
        """Test multilingual prediction with English text"""
        response = client.post('/predict/multilingual', 
                             json={'text': 'I absolutely love this product!'})
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'sentiment' in data
        assert 'confidence' in data
        assert 'language' in data
        assert data['sentiment'] == 'Positive'
    
    def test_predict_multilingual_with_language(self, client):
        """Test multilingual prediction with explicit language"""
        response = client.post('/predict/multilingual', 
                             json={'text': 'This is great!', 'language': 'en'})
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['language'] == 'en'
    
    def test_predict_multilingual_auto_detect(self, client):
        """Test multilingual prediction with auto language detection"""
        response = client.post('/predict/multilingual', 
                             json={'text': 'This is amazing!', 'language': 'auto'})
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'language' in data
    
    def test_predict_multilingual_missing_text(self, client):
        """Test error handling for missing text"""
        response = client.post('/predict/multilingual', json={})
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_predict_multilingual_empty_text(self, client):
        """Test error handling for empty text"""
        response = client.post('/predict/multilingual', json={'text': ''})
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_predict_multilingual_unsupported_language(self, client):
        """Test error handling for unsupported language"""
        response = client.post('/predict/multilingual', 
                             json={'text': 'Hello', 'language': 'xyz'})
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
        assert 'supported_languages' in data


class TestMultilingualBatchPredict:
    """Tests for /predict/multilingual/batch endpoint"""
    
    def test_batch_predict_multilingual(self, client):
        """Test batch multilingual prediction"""
        response = client.post('/predict/multilingual/batch', 
                             json={'texts': ['I am happy', 'I am sad']})
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'results' in data
        assert 'total_count' in data
        assert len(data['results']) == 2
    
    def test_batch_predict_multilingual_missing_texts(self, client):
        """Test error handling for missing texts"""
        response = client.post('/predict/multilingual/batch', json={})
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
    
    def test_batch_predict_multilingual_empty_list(self, client):
        """Test error handling for empty list"""
        response = client.post('/predict/multilingual/batch', json={'texts': []})
        
        assert response.status_code == 400


class TestLanguagesEndpoint:
    """Tests for /languages endpoint"""
    
    def test_get_languages(self, client):
        """Test getting available languages"""
        response = client.get('/languages')
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'languages' in data
        assert 'auto_detect_available' in data
        assert isinstance(data['languages'], list)
        assert len(data['languages']) > 0
        
        # Verify structure of each language entry
        for lang in data['languages']:
            assert 'code' in lang
            assert 'name' in lang
            assert 'flag' in lang
            assert 'available' in lang


class TestLanguageDetection:
    """Tests for /detect-language endpoint"""
    
    
    def test_detect_english(self, client, mock_predictor):
        """Test language detection - mocked"""
        # Configure mock for this specific test
        mock_predictor['obj'].language_detector.detect.return_value = 'en'
        mock_predictor['obj'].language_detector.get_confidence.return_value = {'en': 0.95}
        
        # We need to ensure _multilingual_predictor is set in the app module scope
        with patch('app.api_multilingual._multilingual_predictor', mock_predictor['obj']):
            response = client.post('/detect-language', 
                                 json={'text': 'Hello, how are you doing today?'})
        
            assert response.status_code == 200
            data = response.get_json()
            assert data['detected_language'] == 'en'
            assert 'confidence' in data
    
    def test_detect_french(self, client, mock_predictor):
        """Test language detection - mocked"""
        mock_predictor['obj'].language_detector.detect.return_value = 'fr'
        mock_predictor['obj'].language_detector.get_confidence.return_value = {'fr': 0.95}

        with patch('app.api_multilingual._multilingual_predictor', mock_predictor['obj']):
            response = client.post('/detect-language', 
                                 json={'text': 'Bonjour le monde'})
            
            assert response.status_code == 200
            data = response.get_json()
            assert data['detected_language'] == 'fr'
    
    def test_detect_spanish(self, client, mock_predictor):
        """Test language detection - mocked"""
        mock_predictor['obj'].language_detector.detect.return_value = 'es' 
        
        with patch('app.api_multilingual._multilingual_predictor', mock_predictor['obj']):
            response = client.post('/detect-language', 
                                 json={'text': 'Hola mundo'})
            
            assert response.status_code == 200
            data = response.get_json()
            assert data['detected_language'] == 'es'
    
    def test_detect_missing_text(self, client):
        """Test error handling for missing text"""
        response = client.post('/detect-language', json={})
        
        assert response.status_code == 400
        data = response.get_json()
        assert 'error' in data
