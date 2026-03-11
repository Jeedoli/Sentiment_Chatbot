"""
main.py
────────
FastAPI 애플리케이션 엔트리포인트.

[실행]
  poetry run uvicorn main:app --reload --port 8000

[확인]
  Swagger UI : http://localhost:8000/docs
  ReDoc      : http://localhost:8000/redoc
  헬스체크   : http://localhost:8000/health
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.chat     import router as chat_router
from api.routes.analysis import router as analysis_router
from core.config import get_settings
from core.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작 시 설정 검증, 종료 시 정리."""
    cfg = get_settings()
    logger.info(f"SentiChat API 시작 — model: {cfg.sentiment_model_name}")
    if not cfg.openai_api_key:
        logger.warning("OPENAI_API_KEY 가 설정되지 않았습니다. 챗봇 기능이 비활성화됩니다.")
    yield
    logger.info("SentiChat API 종료")


app = FastAPI(
    title       = "SentiChat API",
    description = (
        "## 고객 문의 감정 분석 기반 AI 챗봇 서비스\n\n"
        "- **감정 분석**: 고객 텍스트의 부정/중립/긍정 분류 (klue/roberta-base fine-tune)\n"
        "- **RAG 챗봇**: 지식베이스 검색 + GPT-4o-mini 기반 자동 응답\n"
        "- **자동 에스컬레이션**: 부정 감정이 강한 경우 상담원 연결 안내"
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── 라우터 등록 ──────────────────────────────────────────────────────────
app.include_router(chat_router,     prefix="/api/v1")
app.include_router(analysis_router, prefix="/api/v1")


@app.get("/", tags=["헬스체크"])
def root():
    """기본 루트. 200을 반환하여 테스트와 헬스체크가 통과되도록 함."""
    return {"status": "ok", "service": "SentiChat API", "version": "1.0.0"}

@app.get("/health", tags=["헬스체크"])
def health():
    """서버 상태 확인"""
    return {"status": "ok", "service": "SentiChat API", "version": "1.0.0"}
