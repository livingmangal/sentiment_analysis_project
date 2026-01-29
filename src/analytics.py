import sqlite3
import datetime
import os

# Ensure the database file is stored safely
DB_NAME = "sentiment_analytics.db"

def init_db():
    """Initialize the analytics database if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Create table for request logs
    c.execute('''CREATE TABLE IF NOT EXISTS request_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  input_text TEXT,
                  predicted_sentiment TEXT,
                  confidence_score REAL,
                  ip_address TEXT)''')
    conn.commit()
    conn.close()

def log_inference(input_text, sentiment, confidence, ip_address="0.0.0.0"):
    """Log a single inference request to the database."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        timestamp = datetime.datetime.utcnow().isoformat()
        
        c.execute("INSERT INTO request_logs (timestamp, input_text, predicted_sentiment, confidence_score, ip_address) VALUES (?, ?, ?, ?, ?)",
                  (timestamp, input_text, sentiment, confidence, ip_address))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging analytics: {e}")

def get_general_analytics():
    """Retrieve usage stats for the admin endpoint."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Total requests
    c.execute("SELECT COUNT(*) FROM request_logs")
    total_requests = c.fetchone()[0]
    
    # Breakdown by sentiment (e.g., how many positive vs negative)
    c.execute("SELECT predicted_sentiment, COUNT(*) FROM request_logs GROUP BY predicted_sentiment")
    sentiment_counts = dict(c.fetchall())
    
    # Get last 5 requests for visibility
    c.execute("SELECT timestamp, predicted_sentiment, confidence_score FROM request_logs ORDER BY id DESC LIMIT 5")
    recent_logs = [{"time": row[0], "sentiment": row[1], "confidence": row[2]} for row in c.fetchall()]
    
    conn.close()
    
    return {
        "total_predictions": total_requests,
        "sentiment_distribution": sentiment_counts,
        "recent_activity": recent_logs
    }

# Initialize DB on module import
init_db()