"""
api/routes/chat.py
───────────────────
챗봇 대화 엔드포인트.

POST /api/v1/chat/           메시지 전송 → 감정 분석 + RAG + LLM 응답
DELETE /api/v1/chat/{sid}   세션 히스토리 초기화
"""

from fastapi import APIRouter

from schema.chat import ChatRequest, ChatResponse
from services import chat_service

router = APIRouter(prefix="/chat", tags=["챗봇"])


@router.post("/", response_model=ChatResponse, summary="고객 문의 처리")
async def send_message(body: ChatRequest) -> ChatResponse:
    """
    고객 메시지를 받아 감정 분석 → RAG 검색 → LLM 응답을 생성합니다.

    - **session_id**: 대화 세션을 구분하는 임의 문자열 (클라이언트가 생성)
    - **message**: 고객 입력 텍스트 (1~1000자)
    - 응답에는 감정 분석 결과, LLM 답변, 참조 출처, 상담원 연결 권고 여부가 포함됩니다.
    """
    return await chat_service.chat(body.session_id, body.message)


@router.delete("/{session_id}", summary="세션 초기화")
def reset_session(session_id: str) -> dict:
    """대화 히스토리를 초기화합니다. (대화 재시작 시 호출)"""
    chat_service.clear_history(session_id)
    return {"message": f"세션 {session_id} 초기화 완료"}
