from app.database import get_all_predictions
from collections import defaultdict
from datetime import datetime

def get_sentiment_trends():
    """
    Processes historical predictions into a format suitable for trend analysis.
    Returns a list of data points with timestamps and sentiment labels.
    """
    predictions = get_all_predictions()
    
    # We want to return data points: {x: timestamp, y: sentiment}
    # For a line chart with "positive"/"negative" on Y-axis, we can provide the raw labels
    # if we configure Chart.js correctly, or map them to numeric values.
    # Mapping: Negative = 0, Positive = 1
    
    trend_data = []
    for pred in predictions:
        trend_data.append({
            'timestamp': pred['timestamp'],
            'sentiment': pred['sentiment'].lower(),
            'text': pred['text'][:50] + '...' if len(pred['text']) > 50 else pred['text']
        })
    
    return trend_data

def get_sentiment_summary():
    """
    Returns a summary of sentiment counts.
    """
    predictions = get_all_predictions()
    summary = defaultdict(int)
    for pred in predictions:
        summary[pred['sentiment'].lower()] += 1
    
    return dict(summary)
