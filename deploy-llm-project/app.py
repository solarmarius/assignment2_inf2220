from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Register Blueprints
from routes.routes import main_bp

app.register_blueprint(main_bp)

if __name__ == "__main__":
    app.run(debug=True)
