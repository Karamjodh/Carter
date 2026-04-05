from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()

@router.get("/health")
def health_check():
    return {
        "status" : "ok",
        "app" : settings.APP_NAME,
        "environment" : settings.ENVIORNMENT,
    }

@router.get("/health/detailed")
def detailed_health():
    return{
        "status" : "ok",
        "app" : settings.APP_NAME,
        "version" : "1.0.0",
        "enviornment" : settings.ENVIORNMENT,
        "services" : {
            "api" : "ok",
            "database" : "not connected yet",
            "ml_pipeline" : "not connected yet",
            "llm" : "not connected yet",
        }
    }