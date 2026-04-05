from fastapi import FastAPI
from app.core.config import settings
from app.api.routes import health

app = FastAPI(
    title = settings.APP_NAME,
    description = "AI-powered customer analytics platform",
    version = "1.0.0",
)
app.include_router(health.router, prefix = "/api/v1", tags = ["Health"])

@app.get("/")
def root():
    return {
        'app' : settings.APP_NAME,
        'environment' : settings.ENVIORNMENT,
        'message' : "Welcome to Carter.ai",
        'docs' : '/docs'
    }