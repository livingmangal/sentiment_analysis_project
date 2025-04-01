# Sentiment Analysis API

A RESTful API for sentiment analysis using a deep learning model (LSTM). The API analyzes text input and returns sentiment predictions with confidence scores.

## Features

- Text sentiment analysis (positive/negative)
- Confidence scores for predictions
- Rate limiting and request validation
- GPU support for faster inference
- Comprehensive error handling
- API documentation

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sentiment_analysis_project.git
cd sentiment_analysis_project
```

2. Create and activate a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the API server:
```bash
python app/api.py
```

2. The API will be available at `http://localhost:5000`

3. Make predictions using curl or any HTTP client:
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was fantastic!"}'
```

Example response:
```json
{
    "sentiment": "Positive",
    "confidence": 0.9234,
    "raw_score": 2.4567
}
```

## API Endpoints

### POST /predict
Analyzes the sentiment of the provided text.

**Request Body:**
```json
{
    "text": "Text to analyze"
}
```

**Response:**
```json
{
    "sentiment": "Positive" | "Negative",
    "confidence": float (0-1),
    "raw_score": float
}
```

**Rate Limits:**
- 10 requests per minute
- 50 requests per hour
- 200 requests per day

## Model Architecture

The sentiment analysis model uses:
- LSTM neural network
- Bidirectional processing
- Dropout regularization
- Word embeddings
- Multi-layer architecture

## Error Handling

The API returns appropriate HTTP status codes:
- 200: Successful prediction
- 400: Invalid request
- 429: Rate limit exceeded
- 500: Server error

## Development

### Project Structure
```
sentiment_analysis_project/
├── app/
│   └── api.py              # Flask API implementation
├── src/
│   ├── model.py           # LSTM model architecture
│   ├── predict.py         # Prediction logic
│   └── preprocessing.py   # Text preprocessing
├── models/
│   ├── lstm_model.pth     # Trained model weights
│   └── preprocessor.pkl   # Preprocessor state
├── tests/                 # Test files
├── notebooks/            # Jupyter notebooks
├── data/                # Training data
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

### Running Tests
```bash
python -m pytest tests/
```

## License

MIT License - see LICENSE file for details
