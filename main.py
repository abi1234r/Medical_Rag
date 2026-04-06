"""
Medical RAG Chatbot — Flask entry point
"""
from flask import Flask
from app.routes import bp

app = Flask(__name__, template_folder="templates", static_folder="static")
app.register_blueprint(bp)

import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)