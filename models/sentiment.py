"""
models/sentiment.py
────────────────────
klue/roberta-base (또는 monologg/kobert) 를 fine-tune한
3-class 감정 분류 모델.

[구조]
  입력: 토큰화된 텍스트 (input_ids, attention_mask)
     ↓
  Transformer Encoder (klue/roberta-base)
     ↓
  [CLS] 토큰 출력 → Dropout(0.3)
     ↓
  Linear(hidden_size → 3)   ← 부정/중립/긍정
     ↓
  Softmax (추론 시)

[왜 klue/roberta-base 인가?]
  - 한국어 특화 사전학습 (뉴스, 위키, SNS 등 60GB 코퍼스)
  - BERT-base와 동일 파라미터 규모(110M) → 개인 환경에서도 학습 가능
  - KLUE 벤치마크에서 NSMC, YNAT 등 태스크 SOTA에 가까운 성능
"""

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


# ── 레이블 매핑 ────────────────────────────────────────────────────────────
LABEL_MAP = {0: "부정", 1: "중립", 2: "긍정"}
N_LABELS  = len(LABEL_MAP)


@dataclass
class SentimentOutput:
    """추론 결과 데이터 클래스"""
    label:     int    # 0=부정 / 1=중립 / 2=긍정
    label_str: str    # "부정" / "중립" / "긍정"
    negative:  float  # 부정 확률 0~1
    neutral:   float  # 중립 확률 0~1
    positive:  float  # 긍정 확률 0~1
    escalate:  bool   # 상담원 연결 권고 여부


class SentimentClassifier(nn.Module):
    """
    AutoModel (RoBERTa/BERT 계열) + 분류 헤드.

    Parameters
    ----------
    model_name : HuggingFace model id 또는 로컬 경로
    num_labels : 분류 클래스 수 (기본 3)
    dropout    : 고정 드롭아웃 비율
    """

    def __init__(
        self,
        model_name: str = "klue/roberta-base",
        num_labels: int = N_LABELS,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden          = self.encoder.config.hidden_size         # 768
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids      : (B, seq_len)
        attention_mask : (B, seq_len)

        Returns
        -------
        logits : (B, num_labels)
        """
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls    = out.last_hidden_state[:, 0, :]   # [CLS] 토큰
        cls    = self.dropout(cls)
        return self.classifier(cls)


# ── 추론 래퍼 ──────────────────────────────────────────────────────────────
class SentimentInference:
    """
    체크포인트를 로드하고 단일/배치 텍스트 추론을 수행.
    매 요청마다 모델을 새로 로드하지 않도록 singleton 패턴 사용.

    Parameters
    ----------
    ckpt_path    : 저장된 state_dict (.pt) 경로
    model_name   : 사전학습 모델 이름 (토크나이저 로드에 필요)
    escalation_t : 부정 확률이 이 값 이상이면 escalate=True
    """

    def __init__(
        self,
        ckpt_path:    str,
        model_name:   str   = "klue/roberta-base",
        max_len:      int   = 128,
        escalation_t: float = 0.7,
    ):
        self.device       = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_len      = max_len
        self.escalation_t = escalation_t

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = SentimentClassifier(model_name=model_name)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"체크포인트를 찾을 수 없습니다: {ckpt_path}\n"
                "먼저  poetry run python scripts/train.py  를 실행하세요."
            )
        self.model.load_state_dict(
            torch.load(ckpt_path, map_location=self.device)
        )
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, text: str) -> SentimentOutput:
        """단일 텍스트 → SentimentOutput"""
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        logits = self.model(input_ids, attention_mask)         # (1, 3)
        probs  = logits.softmax(dim=-1).squeeze(0).cpu().tolist()

        label  = int(torch.argmax(logits, dim=-1).item())
        neg, neu, pos = probs[0], probs[1], probs[2]

        return SentimentOutput(
            label     = label,
            label_str = LABEL_MAP[label],
            negative  = round(neg, 4),
            neutral   = round(neu, 4),
            positive  = round(pos, 4),
            escalate  = neg >= self.escalation_t,
        )

    @torch.no_grad()
    def predict_batch(self, texts: list[str]) -> list[SentimentOutput]:
        """여러 텍스트 배치 추론"""
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
            padding=True,
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        logits = self.model(input_ids, attention_mask)          # (B, 3)
        probs  = logits.softmax(dim=-1).cpu().tolist()
        labels = logits.argmax(dim=-1).cpu().tolist()

        results = []
        for i, (label, prob) in enumerate(zip(labels, probs)):
            neg, neu, pos = prob[0], prob[1], prob[2]
            results.append(SentimentOutput(
                label     = label,
                label_str = LABEL_MAP[label],
                negative  = round(neg, 4),
                neutral   = round(neu, 4),
                positive  = round(pos, 4),
                escalate  = neg >= self.escalation_t,
            ))
        return results
