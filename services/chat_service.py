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

from collections import defaultdict

from langchain_core.messages import AIMessage, HumanMessage

from chains.qa_chain import SentimentChatChain
from core.config import get_settings
from core.logging import logger
from models.sentiment import SentimentOutput
from schema.chat import ChatResponse
from schema.sentiment import SentimentResult, SentimentLabel
from services import rag_service, sentiment_service


# ── 세션 히스토리 in-memory 저장소 ─────────────────────────────────────────
# 실제 서비스에서는 Redis나 DB로 교체하세요.
_history: dict[str, list] = defaultdict(list)

# 체인 singleton
_chat_chain = SentimentChatChain()


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

    # 1. 감정 분석
    sentiment_out = sentiment_service.analyze(message)
    logger.info(
        f"[sentiment] {sentiment_out.label_str} "
        f"(neg={sentiment_out.negative:.2f} pos={sentiment_out.positive:.2f})"
    )

    # 2. RAG 검색
    retrieved = rag_service.retrieve(message)
    context   = "\n\n".join(retrieved) if retrieved else ""
    sources   = [chunk[:60] + "…" for chunk in retrieved]

    # 3. LLM 응답 생성 (히스토리 포함)
    history = _history[session_id]
    answer  = await _chat_chain.ainvoke(
        message   = message,
        sentiment = sentiment_out,
        context   = context,
        history   = history,
    )

    # 4. 히스토리 업데이트 (최대 N턴 유지)
    history.append(HumanMessage(content=message))
    history.append(AIMessage(content=answer))
    max_msgs = cfg.max_history_turns * 2
    if len(history) > max_msgs:
        _history[session_id] = history[-max_msgs:]

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
