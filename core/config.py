"""
core/config.py
───────────────
pydantic-settings 기반 전역 설정 관리.
.env 파일을 자동으로 읽어오며 @lru_cache로 싱글턴 보장.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── OpenAI ─────────────────────────────────────────────────────────
    openai_api_key:    str   = ""
    openai_model:      str   = "gpt-4o-mini"
    openai_temperature: float = 0.3

    # ── 감정 분류 모델 ──────────────────────────────────────────────────
    sentiment_model_name: str = "klue/roberta-base"
    sentiment_model_path: str = "saved_models/sentiment_best.pt"
    num_labels:           int = 3
    train_max_len:        int = 128

    # ── 학습 하이퍼파라미터 ─────────────────────────────────────────────
    train_epochs:       int   = 5
    train_batch_size:   int   = 32
    train_lr:           float = 2e-5
    train_warmup_ratio: float = 0.1

    # ── RAG ────────────────────────────────────────────────────────────
    vectorstore_path:    str = "data/vectorstore"
    knowledge_base_dir:  str = "knowledge_base"
    top_k_retrieval:     int = 3
    embedding_model:     str = "text-embedding-3-small"

    # ── 챗봇 ───────────────────────────────────────────────────────────
    max_history_turns:     int   = 10
    # 부정 감정 확률이 이 임계값 초과 시 상담원 연결 안내
    escalation_threshold:  float = 0.7

    # ── 서버 ───────────────────────────────────────────────────────────
    api_host:     str = "0.0.0.0"
    api_port:     int = 8000
    gradio_port:  int = 7860
    gradio_share: bool = False


@lru_cache
def get_settings() -> Settings:
    """애플리케이션 전역에서 단일 Settings 인스턴스를 반환."""
    return Settings()
