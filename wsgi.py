"""
WSGI entry point for Gunicorn (used by Render and other hosting platforms)
"""
from main import app

if __name__ == "__main__":
    app.run()
