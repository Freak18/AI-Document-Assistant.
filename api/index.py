import sys
from pathlib import Path

# Add backend directory to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
sys.path.append(str(BACKEND_DIR))

# Import the FastAPI app defined in backend/main.py
from main import app

# Vercel expects a variable named `handler` for the serverless function
handler = app
