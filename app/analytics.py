from src.database import get_db_session, Prediction, Feedback, ModelVersion
from sqlalchemy import func, desc, and_
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd
import json

def get_dashboard_stats(start_date=None, end_date=None, variant_id=None, language=None):
    """
    Returns comprehensive statistics for the admin dashboard.
    """
    session = get_db_session()
    
    # Base query
    query = session.query(Prediction)
    
    # Apply filters
    if start_date:
        query = query.filter(Prediction.timestamp >= start_date)
    if end_date:
        query = query.filter(Prediction.timestamp <= end_date)
    if variant_id:
        query = query.filter(Prediction.variant_id == variant_id)
    if language:
        query = query.filter(Prediction.language == language)
    
    predictions = query.all()
    
    if not predictions:
        session.close()
        return {
            "total_predictions": 0,
            "sentiment_distribution": {},
            "trends": [],
            "confidence_distribution": [],
            "drift_metrics": {},
            "language_distribution": {}
        }

    # Convert to list of dicts for processing
    data = [p.to_dict() for p in predictions]
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. Sentiment Distribution
    sentiment_dist = df['sentiment'].value_counts().to_dict()
    
    # 2. Trends (grouped by day)
    df['date'] = df['timestamp'].dt.date
    trends = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0).reset_index()
    trends['date'] = trends['date'].astype(str)
    trends_list = trends.to_dict(orient='records')
    
    # 3. Confidence Distribution (histograms)
    confidence_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    df['conf_bin'] = pd.cut(df['confidence'], bins=confidence_bins)
    conf_dist = df['conf_bin'].value_counts().sort_index().to_dict()
    conf_dist = {str(k): int(v) for k, v in conf_dist.items()}
    
    # 4. Language Distribution
    lang_dist = df['language'].value_counts().to_dict()
    
    # 5. Drift Detection
    # Compare last 100 predictions with previous 1000
    recent_count = min(100, len(df))
    recent_df = df.iloc[-recent_count:]
    baseline_df = df.iloc[:-recent_count] if len(df) > recent_count else df
    
    drift_metrics = {
        "recent_avg_confidence": float(recent_df['confidence'].mean()),
        "baseline_avg_confidence": float(baseline_df['confidence'].mean()) if not baseline_df.empty else 0.0,
        "drift_detected": False
    }
    
    if drift_metrics["baseline_avg_confidence"] > 0:
        confidence_drop = (drift_metrics["baseline_avg_confidence"] - drift_metrics["recent_avg_confidence"]) / drift_metrics["baseline_avg_confidence"]
        if confidence_drop > 0.15: # 15% drop in confidence
            drift_metrics["drift_detected"] = True
            drift_metrics["drift_reason"] = "Significant drop in average prediction confidence."

    # 6. A/B Test Metrics
    ab_metrics = {}
    if 'variant_id' in df.columns:
        variant_groups = df.groupby('variant_id')
        for vid, group in variant_groups:
            ab_metrics[vid] = {
                "count": int(len(group)),
                "avg_confidence": float(group['confidence'].mean()),
                "sentiment_split": group['sentiment'].value_counts(normalize=True).to_dict()
            }

    session.close()
    
    return {
        "total_predictions": len(df),
        "sentiment_distribution": sentiment_dist,
        "trends": trends_list,
        "confidence_distribution": conf_dist,
        "drift_metrics": drift_metrics,
        "language_distribution": lang_dist,
        "ab_metrics": ab_metrics
    }

def export_predictions_csv(start_date=None, end_date=None):
    """
    Generates CSV data for predictions.
    """
    session = get_db_session()
    query = session.query(Prediction)
    if start_date: query = query.filter(Prediction.timestamp >= start_date)
    if end_date: query = query.filter(Prediction.timestamp <= end_date)
    
    predictions = query.all()
    session.close()
    
    if not predictions:
        return "No data found"
        
    df = pd.DataFrame([p.to_dict() for p in predictions])
    return df.to_csv(index=False)

def get_sentiment_trends():
    """Fallback for old calls"""
    stats = get_dashboard_stats()
    return stats['trends']

def get_sentiment_summary():
    """Fallback for old calls"""
    stats = get_dashboard_stats()
    return stats['sentiment_distribution']
