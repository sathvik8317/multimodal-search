"""
Compatibility wrapper.

Keep this file so your existing command still works:
    uvicorn api:app --host 127.0.0.1 --port 8000 --reload

The real FastAPI application now lives in main.py.
"""

from main import app
