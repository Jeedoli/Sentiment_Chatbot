"""
services/chat_service.py
─────────────────────────
대화 세션 관리 + 감정 분석 + RAG 검색 + LLM 응답 생성을 조율하는
핵심 비즈니스 로직 서비스.

[처리 흐름]
  사용자 메시지
      ↓
  1. SentimentService → 감정 분석 (부정/중립/긍정 + 확률)
  ↓
  2. RAGService → 관련 FAQ/정책 검색
  ↓
  3. SentimentChatChain → LLM 응답 생성
  ↓
  4. 세션 히스토리 업데이트
  ↓
  ChatResponse 반환
"""

import asyncio
import functools
import time

try:
    # FastAPI가 설치된 환경에서는 이벤트 루프를 블로킹하지 않기 위해 run_in_threadpool 사용
    from fastapi.concurrency import run_in_threadpool
except ImportError:  # 테스트/CI 환경에서 FastAPI가 없을 수도 있음
    import functools
    async def run_in_threadpool(func, *args, **kwargs):  # type: ignore[misc]
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

try:
    from langchain_core.messages import AIMessage, HumanMessage
    from chains.qa_chain import SentimentChatChain
except ImportError:  # 테스트/CI 환경 또는 langchain 미설치 시
    AIMessage = None  # type: ignore
    HumanMessage = None  # type: ignore
    SentimentChatChain = None  # type: ignore

from core.config import get_settings
from core.logging import logger
from models.sentiment import SentimentOutput
from schema.chat import ChatResponse
from schema.sentiment import SentimentResult, SentimentLabel
from services import rag_service, sentiment_service


# ── 세션 히스토리 in-memory 저장소 ─────────────────────────────────────────
# 실제 서비스에서는 Redis나 DB로 교체하세요.
# 최대 세션 수를 제한해 OOM/DoS 방지 (maxsize=5000)

_MAX_SESSIONS = 5_000
_history: dict[str, list] = {}


def _get_history(session_id: str) -> list:
    """세션 히스토리 반환. 최대 세션 수 초과 시 가장 오래된 세션 제거."""
    if session_id not in _history:
        if len(_history) >= _MAX_SESSIONS:  # noqa: PLR2004
            # 가장 오래된 세션 제거 (dict는 Python 3.7+ 삽입 순서 보장)
            oldest = next(iter(_history))
            del _history[oldest]
            logger.warning(f"세션 수 상한({_MAX_SESSIONS}) 도달 — 오래된 세션 제거: {oldest[:8]}…")
        _history[session_id] = []
    return _history[session_id]

# 체인 singleton (LangChain 미설치 환경에서는 None)
_chat_chain = SentimentChatChain() if SentimentChatChain is not None else None


def _to_schema(so: SentimentOutput) -> SentimentResult:
    """SentimentOutput → Pydantic SentimentResult 변환"""
    return SentimentResult(
        label      = SentimentLabel(so.label),
        label_str  = so.label_str,
        negative   = so.negative,
        neutral    = so.neutral,
        positive   = so.positive,
        escalate   = so.escalate,
    )


async def chat(session_id: str, message: str) -> ChatResponse:
    """
    메인 대화 처리 함수.

    Parameters
    ----------
    session_id : 클라이언트가 전달하는 세션 식별자
    message    : 사용자 입력 텍스트
    """
    cfg = get_settings()
    logger.info(f"[chat] session={session_id} | msg={message[:50]}")

    # 1 & 2: 감정 분석 + RAG 검색 병렬 실행 (서로 독립적이므로 동시에 처리)
    t0 = time.perf_counter()
    sentiment_out, retrieved = await asyncio.gather(
        run_in_threadpool(sentiment_service.analyze, message),
        run_in_threadpool(rag_service.retrieve, message),
    )
    t1 = time.perf_counter()

    logger.info(
        f"[sentiment] {sentiment_out.label_str} "
        f"(neg={sentiment_out.negative:.2f} pos={sentiment_out.positive:.2f}) "
        f"(t={t1-t0:.2f}s)"
    )
    context = "\n\n".join(retrieved) if retrieved else ""
    sources = [chunk[:60] + "…" for chunk in retrieved]
    logger.info(f"[rag] retrieved={len(retrieved)} (t={t1-t0:.2f}s)")

    # 3. LLM 응답 생성 (히스토리 포함)
    if _chat_chain is None:
        raise RuntimeError(
            "LangChain이 설치되어 있지 않아 LLM 응답을 생성할 수 없습니다. "
            "pip install langchain langchain-openai 등을 설치해주세요."
        )

    history = _get_history(session_id)
    t0 = time.perf_counter()
    answer  = await _chat_chain.ainvoke(
        message   = message,
        sentiment = sentiment_out,
        context   = context,
        history   = history,
    )
    t1 = time.perf_counter()
    logger.info(f"[llm] response time {t1-t0:.2f}s")

    # 4. 히스토리 업데이트 (최대 N턴 유지)
    history.append(HumanMessage(content=message))
    history.append(AIMessage(content=answer))
    max_msgs = cfg.max_history_turns * 2
    if len(history) > max_msgs:
        _history[session_id] = history[-max_msgs:]
        history = _history[session_id]

    return ChatResponse(
        session_id = session_id,
        answer     = answer,
        sentiment  = _to_schema(sentiment_out),
        sources    = sources,
        escalate   = sentiment_out.escalate,
    )


def clear_history(session_id: str) -> None:
    """세션 히스토리 초기화 (대화 리셋)"""
    _history.pop(session_id, None)
    logger.info(f"[chat] session={session_id} 히스토리 초기화")
