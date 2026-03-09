"""
api/routes/analysis.py
───────────────────────
감정 분석 전용 엔드포인트.

POST /api/v1/analysis/          단건 텍스트 분석
POST /api/v1/analysis/batch     여러 텍스트 일괄 분석
"""

from fastapi import APIRouter, Depends

from api.deps import get_inference
from models.sentiment import SentimentInference
from schema.sentiment import (
    BatchSentimentRequest,
    SentimentRequest,
    SentimentResult,
    SentimentLabel,
)
from models.sentiment import SentimentOutput


router = APIRouter(prefix="/analysis", tags=["감정 분석"])


def _to_schema(so: SentimentOutput) -> SentimentResult:
    return SentimentResult(
        label     = SentimentLabel(so.label),
        label_str = so.label_str,
        negative  = so.negative,
        neutral   = so.neutral,
        positive  = so.positive,
        escalate  = so.escalate,
    )


@router.post("/", response_model=SentimentResult, summary="단건 텍스트 감정 분석")
def analyze(
    body: SentimentRequest,
    infer: SentimentInference = Depends(get_inference),
) -> SentimentResult:
    """
    텍스트 한 건을 분석하여 부정/중립/긍정 확률을 반환합니다.

    - **label**: 0=부정, 1=중립, 2=긍정
    - **escalate**: 부정 확률이 임계값 초과 시 True (상담원 연결 권고)
    """
    result = infer.predict(body.text)
    return _to_schema(result)


@router.post("/batch", response_model=list[SentimentResult], summary="배치 텍스트 감정 분석")
def analyze_batch(
    body: BatchSentimentRequest,
    infer: SentimentInference = Depends(get_inference),
) -> list[SentimentResult]:
    """여러 텍스트를 한 번에 분석합니다."""
    results = infer.predict_batch(body.texts)
    return [_to_schema(r) for r in results]
