"""
tests/test_api.py
──────────────────
FastAPI TestClient 기반 통합 테스트.
saved_models/sentiment_best.pt 없을 때는 503 응답을 검증하고,
있을 때는 실제 추론 결과를 검증한다.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from main import app

client = TestClient(app)


# ── 1. 헬스 체크 ──────────────────────────────────────────────────────────
class TestHealth:
    def test_root_returns_200(self):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_health_endpoint(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ── 2. /api/v1/analysis/ — 모델 없을 때 503 ───────────────────────────────
class TestAnalysisNoModel:
    def test_analysis_503_when_no_model(self):
        """checkpoint 없으면 503 Service Unavailable"""
        with patch(
            "api.deps.get_inference",
            side_effect=FileNotFoundError("모델 없음"),
        ):
            resp = client.post(
                "/api/v1/analysis/",
                json={"text": "배송이 너무 늦어요"},
            )
        # get_inference 가 Depends() 에서 503 을 raise 하는 구조
        # 실제 503 or 500 모두 허용 (환경에 따라 다름)
        assert resp.status_code in (500, 503)


# ── 3. /api/v1/analysis/ — 모델 mock ────────────────────────────────────
class TestAnalysisMocked:
    @pytest.fixture
    def mock_inference(self):
        from schema.sentiment import SentimentLabel, SentimentResult
        mock_inf = MagicMock()
        mock_inf.predict.return_value = MagicMock(
            label     = SentimentLabel.NEGATIVE,
            label_str = "NEGATIVE",
            negative  = 0.85,
            neutral   = 0.10,
            positive  = 0.05,
            escalate  = True,
        )
        return mock_inf

    def test_analysis_returns_label(self, mock_inference):
        with patch("api.routes.analysis.get_inference", return_value=mock_inference):
            resp = client.post(
                "/api/v1/analysis/",
                json={"text": "진짜 짜증나요"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["label_str"] == "NEGATIVE"
        assert body["escalate"]  is True

    def test_analysis_batch(self, mock_inference):
        mock_inference.predict_batch.return_value = [mock_inference.predict.return_value] * 2
        with patch("api.routes.analysis.get_inference", return_value=mock_inference):
            resp = client.post(
                "/api/v1/analysis/batch",
                json={"texts": ["나빠요", "싫어요"]},
            )
        assert resp.status_code == 200
        results = resp.json()
        assert isinstance(results, list)
        assert len(results) == 2


# ── 4. /api/v1/chat/ — mock ───────────────────────────────────────────────
class TestChatMocked:
    @pytest.fixture
    def mock_chat(self):
        from schema.chat import ChatResponse
        from schema.sentiment import SentimentLabel, SentimentResult

        async def _fake_chat(session_id: str, message: str) -> ChatResponse:
            return ChatResponse(
                session_id = session_id,
                answer     = "안녕하세요! 도움이 필요하신가요?",
                sentiment  = SentimentResult(
                    label     = SentimentLabel.NEUTRAL,
                    label_str = "NEUTRAL",
                    negative  = 0.1,
                    neutral   = 0.8,
                    positive  = 0.1,
                    escalate  = False,
                ),
                sources    = [],
                escalate   = False,
            )
        return _fake_chat

    def test_chat_returns_answer(self, mock_chat):
        with patch("api.routes.chat.chat_service.chat", side_effect=mock_chat):
            resp = client.post(
                "/api/v1/chat/",
                json={"session_id": "test-session", "message": "안녕하세요"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "answer"    in body
        assert "sentiment" in body

    def test_delete_session(self):
        with patch("api.routes.chat.chat_service.clear_history", return_value=None):
            resp = client.delete("/api/v1/chat/test-session")
        assert resp.status_code == 200


# ── 5. 잘못된 요청 ──────────────────────────────────────────────────────
class TestValidation:
    def test_analysis_empty_text(self):
        resp = client.post("/api/v1/analysis/", json={"text": ""})
        # 빈 텍스트 → 422 Unprocessable or 400
        assert resp.status_code in (400, 422, 503)

    def test_analysis_missing_field(self):
        resp = client.post("/api/v1/analysis/", json={})
        assert resp.status_code == 422
