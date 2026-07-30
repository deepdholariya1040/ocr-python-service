from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.services import ocr_engine

router = APIRouter(tags=["health"])


@router.get("/health")
@router.get("/")
def health(settings: Settings = Depends(get_settings)) -> dict:
    return {
        "status": "ok",
        "service": settings.SERVICE_NAME,
        "environment": settings.ENVIRONMENT,
        "geminiConfigured": bool(settings.GEMINI_API_KEY),
        "ocrEngineReady": ocr_engine.is_ready(),
    }
