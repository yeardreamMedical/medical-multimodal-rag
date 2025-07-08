from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str
    PINECONE_TEXT_INDEX_NAME: str
    PINECONE_IMAGE_INDEX_NAME: str

    # Gemini
    GEMINI_API_KEY: str
    OPENAI_API_KEY: str

    class Config:
        env_file = ".env"

settings = Settings()
