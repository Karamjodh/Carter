from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    APP_NAME : str = "Carter.ai"
    ENVIORNMENT : str = "development"
    ANTHROPIC_API_KEY : str = ""

    class Config:
        env_file = ".env",
        extra = "ignore"

settings = Settings()