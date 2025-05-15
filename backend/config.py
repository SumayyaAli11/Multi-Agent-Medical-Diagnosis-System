from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "llama3-8b-8192"
    
    class Config:
        env_file = ".env"

settings = Settings()