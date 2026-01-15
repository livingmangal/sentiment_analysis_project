import sqlite3
import os
from datetime import datetime

# Get the base directory for the database
basedir = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(basedir, 'sentiment.db')

def init_db():
    """Initialize the database and create the predictions table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(text, sentiment, confidence, timestamp=None):
    """Save a prediction to the database with an optional timestamp."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if not timestamp:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute(
        'INSERT INTO predictions (text, sentiment, confidence, timestamp) VALUES (?, ?, ?, ?)',
        (text, sentiment, confidence, timestamp)
    )
    conn.commit()
    conn.close()

def get_all_predictions():
    """Retrieve all predictions from the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions ORDER BY timestamp ASC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def clear_all_predictions():
    """Delete all predictions from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM predictions')
    conn.commit()
