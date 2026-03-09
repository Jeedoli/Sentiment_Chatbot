"""
tests/test_sentiment.py
────────────────────────
모델 체크포인트 없이 실행 가능한 단위 테스트.
학습 시 실제 모델을 로드하므로 CI에서는 SKIP.
"""

import pytest
import torch

from schema.sentiment import SentimentLabel, SentimentResult


# ── 1. SentimentLabel ──────────────────────────────────────────────────────
class TestSentimentLabel:
    def test_values(self):
        assert SentimentLabel.NEGATIVE == 0
        assert SentimentLabel.NEUTRAL  == 1
        assert SentimentLabel.POSITIVE == 2

    def test_from_int(self):
        assert SentimentLabel(0) is SentimentLabel.NEGATIVE
        assert SentimentLabel(2) is SentimentLabel.POSITIVE

    def test_names(self):
        assert SentimentLabel.NEGATIVE.name == "NEGATIVE"


# ── 2. SentimentResult ────────────────────────────────────────────────────
class TestSentimentResult:
    def _make(self, label: int = 0, neg=0.8, neu=0.1, pos=0.1):
        return SentimentResult(
            label     = SentimentLabel(label),
            label_str = SentimentLabel(label).name,
            negative  = neg,
            neutral   = neu,
            positive  = pos,
            escalate  = neg >= 0.7,
        )

    def test_probabilities_sum(self):
        r = self._make()
        assert abs((r.negative + r.neutral + r.positive) - 1.0) < 1e-4

    def test_escalate_true_when_high_negative(self):
        r = self._make(neg=0.75)
        assert r.escalate is True

    def test_escalate_false_when_low_negative(self):
        r = self._make(label=2, neg=0.05, neu=0.10, pos=0.85)
        assert r.escalate is False

    def test_label_field(self):
        r = self._make(label=1, neg=0.1, neu=0.8, pos=0.1)
        assert r.label == SentimentLabel.NEUTRAL


# ── 3. SentimentClassifier 구조 테스트 (CPU, 가중치 없이) ─────────────────
class TestSentimentClassifierStructure:
    def test_forward_shape(self):
        """학습 없이 random weight 으로 out shape 검증"""
        from models.sentiment import SentimentClassifier

        model = SentimentClassifier(num_labels=3)
        model.eval()

        # 더미 토큰 (batch=2, seq=16)
        dummy_ids  = torch.ones(2, 16, dtype=torch.long)
        dummy_mask = torch.ones(2, 16, dtype=torch.long)

        with torch.no_grad():
            out = model(input_ids=dummy_ids, attention_mask=dummy_mask)

        assert out.shape == (2, 3), f"expected (2,3), got {out.shape}"

    def test_dropout_inactive_in_eval(self):
        """eval 모드에서 두 번 forward 결과가 동일해야 함"""
        from models.sentiment import SentimentClassifier

        model = SentimentClassifier(num_labels=3)
        model.eval()

        ids  = torch.ones(1, 8, dtype=torch.long)
        mask = torch.ones(1, 8, dtype=torch.long)

        with torch.no_grad():
            out1 = model(input_ids=ids, attention_mask=mask)
            out2 = model(input_ids=ids, attention_mask=mask)

        assert torch.allclose(out1, out2)


# ── 4. SentimentDataset ───────────────────────────────────────────────────
class TestSentimentDataset:
    @pytest.fixture
    def sample_csv(self, tmp_path):
        import pandas as pd
        df = pd.DataFrame({
            "text" : ["정말 좋아요", "그냥 그래요", "너무 나빠요"],
            "label": [2, 1, 0],
        })
        p = tmp_path / "sample.csv"
        df.to_csv(p, index=False)
        return str(p)

    def test_dataset_len(self, sample_csv):
        from scripts.train import SentimentDataset
        ds = SentimentDataset(sample_csv, max_length=32)
        assert len(ds) == 3

    def test_dataset_item_keys(self, sample_csv):
        from scripts.train import SentimentDataset
        ds = SentimentDataset(sample_csv, max_length=32)
        item = ds[0]
        assert "input_ids"      in item
        assert "attention_mask" in item
        assert "labels"         in item
