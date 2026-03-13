# SentiChat — 감정 분석 기반 고객 서비스 AI 챗봇

> klue/roberta-base 파인튜닝 + LangChain RAG 파이프라인으로 고객 문의의 감정을 실시간 분석하고,  
> 감정 상태에 맞는 응답을 자동 생성하는 서비스입니다.

---

## 🗂 프로젝트 구조

```
sentiment_chatbot/
├── api/
│   ├── deps.py              # FastAPI 의존성 주입 (모델 로드, DB 세션 등)
│   └── routes/
│       ├── analysis.py      # POST /api/v1/analysis/ — 감정 분석 API
│       └── chat.py          # POST /api/v1/chat/    — 챗봇 대화 API
├── chains/
│   └── qa_chain.py          # LangChain LCEL 체인 (감정 인식 시스템 프롬프트)
├── core/
│   ├── config.py            # pydantic-settings 전역 설정
│   └── logging.py           # loguru 로거 초기화
├── knowledge_base/
│   └── faq.txt              # RAG 참조 문서 (배송/교환/환불 FAQ)
├── models/
│   └── sentiment.py         # SentimentClassifier (nn.Module) + SentimentInference
├── schema/
│   ├── chat.py              # ChatRequest / ChatResponse Pydantic 모델
│   └── sentiment.py         # SentimentLabel / SentimentResult Pydantic 모델
├── scripts/
│   ├── preprocess.py        # NSMC 또는 로컬 CSV → train/val/test.csv
│   ├── train.py             # 파인튜닝 학습 루프 (AdamW + warmup + 조기 종료)
│   └── build_vectorstore.py # knowledge_base/*.txt → FAISS 인덱스 생성
├── services/
│   ├── chat_service.py      # 감정 분석 → RAG 검색 → LLM 응답 오케스트레이션
│   ├── rag_service.py       # FAISS 로드 및 similarity_search 래퍼
│   └── sentiment_service.py # SentimentInference 싱글턴 래퍼
├── tests/
│   ├── test_api.py          # FastAPI TestClient 통합 테스트
│   └── test_sentiment.py    # 모델·데이터셋 단위 테스트
├── app.py                   # Gradio 데모 UI
├── main.py                  # FastAPI 진입점 (lifespan, CORS, 라우터 등록)
└── pyproject.toml
```

---

## ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| **감정 분류** | 부정 / 중립 / 긍정 3-class 분류 (klue/roberta-base 파인튜닝) |
| **RAG 응답** | FAISS 벡터스토어에서 관련 FAQ를 검색하여 LLM 컨텍스트로 주입 |
| **에스컬레이션** | 부정 확률 ≥ 70% 시 상담원 연결 자동 권고 (`escalate=True`) |
| **세션 관리** | in-memory 히스토리 (최근 10턴), session_id 기반 멀티 세션 |
| **REST API** | `/api/v1/analysis/`, `/api/v1/analysis/batch`, `/api/v1/chat/` |
| **Gradio UI** | 감정 확률 바 시각화 + 에스컬레이션 알림 포함 데모 인터페이스 |

---

## 🔧 기술 스택

- **모델**: [klue/roberta-base](https://huggingface.co/klue/roberta-base) (110M params, 한국어 특화)
- **학습**: PyTorch 2.x, Hugging Face Transformers / Datasets
- **데이터**: [NSMC](https://huggingface.co/datasets/nsmc) 또는 로컬 CSV (pandas 전처리)
- **LLM 파이프라인**: LangChain 0.3 LCEL — `ChatPromptTemplate | ChatOpenAI | StrOutputParser`
- **벡터 검색**: FAISS + OpenAI text-embedding-3-small
- **백엔드**: FastAPI 0.115 + Pydantic v2 + pydantic-settings
- **UI**: Gradio 4.x
- **로깅**: loguru (파일 rotation 30일)

---

## 🔧 Workflow

이 프로젝트는 아래 5단계로 구성됩니다.

1. **데이터 준비**
   - AI Hub, NSMC, 로컬 CSV/Excel/ZIP 등 다양한 형태의 데이터를
     `scripts/preprocess.py`로 읽어 `data/processed/train.csv`,
     `val.csv`, `test.csv`로 변환합니다.
   - `--csv_path`에 디렉터리 또는 ZIP 경로를 지정하면 자동으로 내부
     파일을 모두 찾아서 병합합니다.
   - AI Hub 같은 경우, `data/aihub_dataset/Training` 아래에
     `01.원천데이터`/`02.라벨링데이터` 구조만 맞추면 자동 병합됩니다.

2. **모델 학습**
   - `scripts/train.py`를 실행하면 `train.csv`/`val.csv`를 읽어
     `klue/roberta-base` 기반 분류 모델을 fine‑tune 합니다.
   - 학습 중 `runs/` 디렉터리에 TensorBoard 로그가 자동 생성됩니다.
     ```bash
     tensorboard --logdir runs
     ```
   - 학습 중 **val F1이 개선될 때마다** `saved_models/sentiment_best.pt`가
     갱신되고, 마지막 에폭 모델은 `saved_models/sentiment_last.pt`에 저장됩니다.

3. **RAG 벡터스토어 구축**
   - `knowledge_base/` 폴더의 `.txt` 문서를 FAISS 벡터스토어로 변환합니다.
   - `scripts/build_vectorstore.py`를 실행하면 `data/vectorstore/`가 생성됩니다.

4. **서버 실행 (API / UI)**
   - FastAPI: `poetry run uvicorn main:app --reload`
   - Gradio 데모: `poetry run python app.py`

5. **추론 / 챗봇**
   - 클라이언트에서 입력된 텍스트를 감정분석하고, 필요시 RAG 검색 결과를
     합쳐서 GPT 모델에 전달합니다.
   - 부정 비율이 70% 이상인 경우 `escalate=True`로 상담원 연결 유도 기능을 사용합니다.

---

## 🚀 빠른 시작

```bash
# 0. Python 버전 확인 & pyenv 설정
# 이 프로젝트는 macOS Apple Silicon 기준 Python 3.11.x(예: 3.11.9)를 사용.
# pyenv 사용자의 경우 `.python-version` 파일이 루트에 있으므로
#   pyenv install && cd 프로젝트 디렉터리만으로 자동 활성화.
# 다른 버전일 경우 `pyenv local 3.11.9`로 맞추고, 시스템 Python이면
#   적절한 3.11.x를 설치한 뒤 `poetry env use python3.11`을 실행할 것!

# 1. 의존성 설치
poetry install
# torch는 플랫폼별로 별도 설치(아래 추가 설명 참조)
pip install torch torchvision torchaudio   # macOS

# 2. 환경 변수 설정
cp .env.example .env
# .env 에서 OPENAI_API_KEY 입력

# 3. 샘플 데이터로 테스트 학습 (10 에폭, ~2분)
poetry run python scripts/preprocess.py --source sample
poetry run python scripts/train.py --epochs 10

# 4. RAG 벡터스토어 빌드
poetry run python scripts/build_vectorstore.py

# 5-a. FastAPI 서버 실행
poetry run uvicorn main:app --reload

# 5-b. Gradio 데모 UI 실행
poetry run python app.py
```

---

## 📡 API 예시

```bash
# 감정 분석
curl -X POST http://localhost:8000/api/v1/analysis/ \
     -H "Content-Type: application/json" \
     -d '{"text": "배송이 너무 늦어요. 정말 실망입니다."}'

# 챗봇 대화
curl -X POST http://localhost:8000/api/v1/chat/ \
     -H "Content-Type: application/json" \
     -d '{"session_id": "user-001", "message": "환불은 어떻게 하나요?"}'
```

---

## 🧪 테스트 실행

```bash
poetry run pytest tests/ -v
```

---

## 📈 모델 성능 (NSMC 기준 참고치)

| 지표 | 값 |
|------|----|
| Accuracy | ~91% |
| F1-macro | ~90% |
| 학습 시간 | ~15분 (T4 GPU) |

> 실제 성능은 데이터 규모와 하이퍼파라미터에 따라 달라집니다.

---

## 🏗 아키텍처 다이어그램

```
사용자 입력
    │
    ▼
FastAPI  (/api/v1/chat/)
    │
    ├─► SentimentService ─► klue/roberta-base ─► 감정 레이블 + 확률
    │
    ├─► RAGService ─► FAISS 벡터스토어 ─► 관련 FAQ 청크 k개
    │
    └─► SentimentChatChain (LCEL)
            SystemPrompt (감정별 응대 방침)
            + 감정 결과 + FAQ 컨텍스트 + 대화 히스토리
            │
            ▼
          ChatOpenAI (GPT-4o-mini)
            │
            ▼
          응답 반환 (escalate 여부 포함)
```

---

## 📝 라이선스

MIT
