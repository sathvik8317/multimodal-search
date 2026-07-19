"""
Store-memory FastAPI entry point.

Keep this file so the existing command still works:
    uvicorn store_memory:app --host 127.0.0.1 --port 8001 --reload

The actual route and business logic live in the store/ package.
"""

from fastapi import FastAPI

from store.routes import router as store_router


app = FastAPI()
app.include_router(store_router)