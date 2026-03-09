"""
schema/chat.py
───────────────
챗봇 대화 관련 Pydantic 스키마.
"""

from pydantic import BaseModel, Field
from schema.sentiment import SentimentResult


class ChatMessage(BaseModel):
    """단일 대화 메시지"""
    role:    str = Field(..., description="user | assistant")
    content: str = Field(..., description="메시지 내용")


class ChatRequest(BaseModel):
    """챗봇 요청 본문"""
    session_id: str  = Field(..., description="세션 식별자 (프론트/클라이언트가 생성)")
    message:    str  = Field(..., min_length=1, max_length=1000, description="사용자 입력 메시지")


class ChatResponse(BaseModel):
    """챗봇 응답 본문"""
    session_id:  str             = Field(..., description="요청과 동일한 세션 ID")
    answer:      str             = Field(..., description="챗봇 최종 응답 텍스트")
    sentiment:   SentimentResult = Field(..., description="입력 메시지의 감정 분석 결과")
    sources:     list[str]       = Field(default_factory=list, description="참조한 FAQ/지식베이스 출처")
    escalate:    bool            = Field(False, description="상담원 연결 권고 여부")
