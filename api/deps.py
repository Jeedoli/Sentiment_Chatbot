"""
api/deps.py
────────────
FastAPI 의존성 주입 모음.
모델/서비스 singleton을 라우터에서 안전하게 주입합니다.
"""

from fastapi import HTTPException
from services.sentiment_service import get_sentiment_service
from models.sentiment import SentimentInference


def get_inference() -> SentimentInference:
    """감정 분류 인스턴스 반환. 모델 미학습 시 503."""
    try:
        return get_sentiment_service()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
