"""
chains/qa_chain.py
───────────────────
LangChain LCEL 기반 감정 인식 챗봇 체인.

[흐름]
  사용자 메시지 + 감정 분석 결과 + 검색된 FAQ 문서
      ↓
  ChatPromptTemplate (System + Human)
      ↓
  ChatOpenAI (gpt-4o-mini)
      ↓
  StrOutputParser → 최종 응답 텍스트

[설계 원칙]
  - System 프롬프트: 상담원 페르소나, 감정별 응대 방식 지정
  - Human 프롬프트: 메시지 + 감정 레이블 + 검색 컨텍스트 + 대화히스토리
  - temperature=0.3: 상담 챗봇은 일관된 응답 필요, 낮게 유지
  - 부정 감정이 감지되면 공감 어조 + 상담원 연결 안내 포함
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from core.config import get_settings
from models.sentiment import SentimentOutput


# ── 시스템 프롬프트 ────────────────────────────────────────────────────────
_SYSTEM = """\
당신은 친절하고 공감 능력이 뛰어난 고객 서비스 AI 상담원입니다.
고객의 감정 상태를 파악하고 적절한 방식으로 응대합니다.

## 감정별 응대 방침
- 부정(화남/불만): 먼저 고객의 감정에 충분히 공감하고, 문제 해결 방안을 우선 제시합니다.
  부정 확률이 70% 이상이면 응답 마지막에 "전문 상담원 연결을 원하시면 #상담원연결 을 입력해 주세요." 를 추가합니다.
- 중립: 필요한 정보를 명확하고 간결하게 제공합니다.
- 긍정: 밝고 따뜻한 어조로 응답하며 추가 도움 여부를 확인합니다.

## 응답 규칙
- 아래 [참고 문서]에 관련 정보가 있다면 반드시 참고하여 구체적인 답변을 제공하세요.
- 모르는 내용은 솔직히 "확인 후 안내해 드리겠습니다." 라고 말하세요.
- 응답은 자연스러운 한국어로, 200자 이내로 간결하게 작성하세요.
- 이모지는 긍정/중립 응답에만 적절히 사용하세요.
"""

_HUMAN = """\
[고객 감정 분석]
- 감정: {label_str} (부정 {neg:.0%} / 중립 {neu:.0%} / 긍정 {pos:.0%})

[참고 문서]
{context}

[고객 문의]
{message}
"""


def _build_chain():
    cfg = get_settings()
    llm = ChatOpenAI(
        model       = cfg.openai_model,
        temperature = cfg.openai_temperature,
        api_key     = cfg.openai_api_key,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM),
        MessagesPlaceholder(variable_name="history"),
        ("human", _HUMAN),
    ])
    return prompt | llm | StrOutputParser()


# ── 공개 API ──────────────────────────────────────────────────────────────
class SentimentChatChain:
    """
    감정 분석 결과와 RAG 컨텍스트를 포함한 챗봇 체인.

    Parameters
    ----------
    history : 이전 대화 이력 (HumanMessage/AIMessage 리스트)
    """

    def __init__(self):
        self._chain = _build_chain()

    def invoke(
        self,
        message:   str,
        sentiment: SentimentOutput,
        context:   str,
        history:   list | None = None,
    ) -> str:
        """동기 호출"""
        return self._chain.invoke({
            "history":   history or [],
            "label_str": sentiment.label_str,
            "neg":       sentiment.negative,
            "neu":       sentiment.neutral,
            "pos":       sentiment.positive,
            "context":   context or "관련 참고 문서가 없습니다.",
            "message":   message,
        })

    async def ainvoke(
        self,
        message:   str,
        sentiment: SentimentOutput,
        context:   str,
        history:   list | None = None,
    ) -> str:
        """비동기 호출 (FastAPI async endpoint 용)"""
        return await self._chain.ainvoke({
            "history":   history or [],
            "label_str": sentiment.label_str,
            "neg":       sentiment.negative,
            "neu":       sentiment.neutral,
            "pos":       sentiment.positive,
            "context":   context or "관련 참고 문서가 없습니다.",
            "message":   message,
        })
