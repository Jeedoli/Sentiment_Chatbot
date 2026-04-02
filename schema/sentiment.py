"""
schema/sentiment.py
────────────────────
감정 분석 관련 Pydantic 스키마.
"""

from enum import IntEnum
from pydantic import BaseModel, Field, field_validator


class SentimentLabel(IntEnum):
    """감정 레이블 정수 → 문자열 매핑"""
    NEGATIVE = 0
    NEUTRAL  = 1
    POSITIVE = 2


LABEL_KR = {
    SentimentLabel.NEGATIVE: "부정 😠",
    SentimentLabel.NEUTRAL:  "중립 😐",
    SentimentLabel.POSITIVE: "긍정 😊",
}


class SentimentResult(BaseModel):
    """감정 분류 결과 단건"""
    label:       SentimentLabel = Field(..., description="감정 레이블 정수 (0=부정,1=중립,2=긍정)")
    label_str:   str            = Field(..., description="감정 라벨 한글 문자열")
    negative:    float          = Field(..., ge=0, le=1, description="부정 확률")
    neutral:     float          = Field(..., ge=0, le=1, description="중립 확률")
    positive:    float          = Field(..., ge=0, le=1, description="긍정 확률")
    escalate:    bool           = Field(False, description="상담원 연결 권고 여부")


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="분석할 텍스트")


class BatchSentimentRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="배치 분석할 텍스트 목록 (최대 50건)",
    )

    @field_validator("texts")
    @classmethod
    def validate_text_lengths(cls, v: list[str]) -> list[str]:
        for i, text in enumerate(v):
            if len(text) < 1:
                raise ValueError(f"texts[{i}]: 텍스트는 1자 이상이어야 합니다")
            if len(text) > 1000:
                raise ValueError(f"texts[{i}]: 텍스트는 1000자 이하여야 합니다 (현재 {len(text)}자)")
        return v
