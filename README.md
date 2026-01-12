# Sentiment Analysis Web Application

A modern web application for analyzing the sentiment of text using deep learning. Built with Flask, PyTorch, and a beautiful responsive UI.

![Sentiment Analysis App](app/static/images/background.webp)

## 🚀 Live Demo

🔗 **Live Application:** [Click here to open](https://sentiment-analysis-project-oyz1.onrender.com/)


## Features

- **Text Sentiment Analysis**: Analyze any text to determine if it's positive or negative
- **Confidence Scores**: Get confidence levels for each prediction
- **Modern UI**: Clean, responsive interface with beautiful animations
- **Real-time Analysis**: Instant results with visual feedback
- **Cross-Platform**: Works on desktop and mobile devices
- **Deployment Ready**: Configured for deployment on Render

## Technology Stack

- **Backend**: Flask, Python
- **Machine Learning**: PyTorch, LSTM neural network
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Render, Gunicorn

## Project Structure

```
sentiment_analysis_project/
├── app/
│   ├── api.py              # Flask API implementation
│   ├── static/
│   │   ├── css/            # Stylesheets
│   │   ├── js/             # JavaScript files
│   │   └── images/         # Images including background
│   └── templates/
│       └── index.html      # Main HTML template
├── src/
│   ├── model.py            # LSTM model architecture
│   ├── predict.py          # Prediction logic
│   ├── preprocessing.py    # Text preprocessing
│   ├── dataset.py          # Dataset handling
│   ├── train.py            # Model training script
│   └── evaluate.py         # Model evaluation
├── models/
│   ├── lstm_model.pth      # Trained model weights
│   └── preprocessor.pkl    # Preprocessor state
├── data/
│   └── train.csv           # Training data
├── main.py                 # Application entry point
├── Procfile                # Process file for Render
├── render.yaml             # Render deployment configuration
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sentiment_analysis_project.git
cd sentiment_analysis_project
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Locally

1. Start the Flask server:
```bash
python main.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Enter text in the input field and click "Analyze Sentiment"

### API Usage

You can also use the API directly:

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

## Deployment

This application is configured for deployment on Render:

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn --bind 0.0.0.0:$PORT main:app`

## Model Details

The sentiment analysis model uses:
- LSTM neural network for sequence processing
- Bidirectional processing for better context understanding
- Dropout regularization to prevent overfitting
- Word embeddings for semantic representation

## Limitations

- Currently only supports binary classification (Positive/Negative)
- No neutral sentiment classification
- Limited to English text
- Requires sufficient text length for accurate analysis

## Future Improvements

- Add neutral sentiment classification
- Support for multiple languages
- Enhanced visualization of sentiment scores
- User authentication and history tracking
- API key management for production use

## License

Copyright rights reserved to MS, 2024

## Acknowledgements

- Built using Flask and PyTorch
- Inspired by the need for accessible sentiment analysis tools
