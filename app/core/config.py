from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
class Settings(BaseSettings):
    APP_NAME : str = "CarterX.ai"
    ENVIORNMENT : str = "development"
    ANTHROPIC_API_KEY : Optional[str] = None
    GEMINI_API_KEY : Optional[str] = None
    OPENAI_API_KEY : Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    DEFAULT_LLM : str = "groq"
    MAX_UPLOAD_SIZE_MB : int = 50
    MIN_ROWS_REQUIRED : int = 100
    ALLOWED_EXTENSIONS : list[str] = [".csv", ".xlsx"]
    model_config = SettingsConfigDict(
        env_file = ".env",
        extra = "ignore"
    )
settings = Settings()