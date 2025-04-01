from app.api import app
import os

# This is needed for gunicorn to find the app
application = app

if __name__ == '__main__':
    # This is only used when running directly with Python
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
