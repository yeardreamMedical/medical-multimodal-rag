from fastapi import FastAPI
from .api import router as api_router

app = FastAPI(
    title="Medicube API",
    description="API for generating medical questions using RAG and Gemini-1.5-Pro.",
    version="1.0.0",
)

app.include_router(api_router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Medicube API"}
