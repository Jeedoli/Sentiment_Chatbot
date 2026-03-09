"""
services/sentiment_service.py
──────────────────────────────
SentimentInference 를 singleton으로 관리하는 서비스 레이어.
FastAPI의 의존성 주입(deps.py)에서 호출합니다.
"""

from functools import lru_cache

from core.config import get_settings
from models.sentiment import SentimentInference, SentimentOutput


@lru_cache
def get_sentiment_service() -> SentimentInference:
    """최초 1회만 모델을 로드하고 이후 캐싱된 인스턴스를 반환."""
    cfg = get_settings()
    return SentimentInference(
        ckpt_path    = cfg.sentiment_model_path,
        model_name   = cfg.sentiment_model_name,
        max_len      = cfg.train_max_len,
        escalation_t = cfg.escalation_threshold,
    )


def analyze(text: str) -> SentimentOutput:
    return get_sentiment_service().predict(text)


def analyze_batch(texts: list[str]) -> list[SentimentOutput]:
    return get_sentiment_service().predict_batch(texts)
