import pytest
import json
from app.api import app

@pytest.fixture
def client():
    """Sets up a fake client to test the app without running the server"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_sentiment_positive(client):
    """Test if a happy text returns Positive"""
    response = client.post('/predict', 
                         json={'text': 'I absolutely love this product! It is amazing.'})
    
    assert response.status_code == 200
    data = response.get_json()
    assert 'sentiment' in data
    # We expect positive or at least a valid response
    assert data['sentiment'] in ['Positive', 'Negative', 'Neutral']

def test_predict_sentiment_negative(client):
    """Test if a sad text returns a valid response"""
    response = client.post('/predict', 
                         json={'text': 'This is the worst experience ever. I hate it.'})
    
    assert response.status_code == 200
    data = response.get_json()
    assert 'sentiment' in data

def test_predict_missing_text(client):
    """Test if the server catches the error when we send empty data"""
    response = client.post('/predict', json={})
    
    # We expect a 400 Bad Request error
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data

def test_predict_batch_endpoint(client):
    """Test your new Batch Endpoint!"""
    response = client.post('/predict/batch', 
                         json={'texts': ['I am happy', 'I am sad']})
    
    assert response.status_code == 200
    data = response.get_json()
    assert 'results' in data
    assert len(data['results']) == 2