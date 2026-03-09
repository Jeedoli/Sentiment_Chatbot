"""
app.py
───────
Gradio 기반 챗봇 데모 UI.

[실행]
  poetry run python app.py

[접속]
  http://localhost:7860

[UI 구성]
  ┌──────────────────────────────────────────────────────────┐
  │    SentiChat 🤖 — 감정 분석 기반 고객 서비스 챗봇            │
  ├──────────────────────────────────────────────────────────┤
  │  대화 입력창 + 전송 버튼                                    │
  │  대화 히스토리 (ChatInterface)                             │
  ├──────────────────────────────────────────────────────────┤
  │  [감정 분석 결과] 탭    → 부정/중립/긍정 확률 바              │
  │  [참조 출처] 탭         → RAG가 참조한 FAQ 문서              │
  └──────────────────────────────────────────────────────────┘
"""

import os
import uuid

import gradio as gr

from core.config import get_settings
from core.logging import logger

cfg        = get_settings()
_session_id = str(uuid.uuid4())   # 데모용 단일 세션


# ── 감정 이모지 매핑 ──────────────────────────────────────────────────────
_EMOJI = {0: "😠 부정", 1: "😐 중립", 2: "😊 긍정"}


def _sentiment_bar(neg: float, neu: float, pos: float) -> str:
    """텍스트 기반 확률 바 생성"""
    def bar(v: float, color: str) -> str:
        filled = int(v * 20)
        return f"{color}{'█' * filled}{'░' * (20 - filled)} {v:.0%}"

    return (
        f"부정: {bar(neg, '🔴')}\n"
        f"중립: {bar(neu, '🟡')}\n"
        f"긍정: {bar(pos, '🟢')}"
    )


# ── 챗봇 콜백 ─────────────────────────────────────────────────────────────
def respond(message: str, history: list[list]) -> tuple[str, list[list], str, str]:
    """
    Gradio ChatInterface 콜백.

    Returns
    -------
    ("", history, sentiment_text, sources_text)
    """
    if not message.strip():
        return "", history, "", ""

    # 모델 미학습 시 안내
    try:
        from services import chat_service
        import asyncio
        result = asyncio.run(chat_service.chat(_session_id, message))
    except FileNotFoundError:
        tip = (
            "⚠ 학습된 모델이 없습니다.\n\n"
            "먼저 아래 명령어를 실행해 주세요:\n"
            "  1. poetry run python scripts/preprocess.py --source sample\n"
            "  2. poetry run pip install torch\n"
            "  3. poetry run python scripts/train.py\n"
        )
        history.append([message, tip])
        return "", history, "", ""
    except Exception as e:
        logger.error(f"chat error: {e}")
        history.append([message, f"오류가 발생했습니다: {e}"])
        return "", history, "", ""

    # 대화 히스토리 업데이트
    escalate_note = "\n\n---\n📞 전문 상담원 연결을 원하시면 **#상담원연결** 을 입력해 주세요." if result.escalate else ""
    history.append([message, result.answer + escalate_note])

    # 감정 분석 텍스트
    s = result.sentiment
    sentiment_text = (
        f"**{_EMOJI.get(s.label, s.label_str)}**  "
        f"(부정 확률: {s.negative:.0%})\n\n"
        + _sentiment_bar(s.negative, s.neutral, s.positive)
        + ("\n\n🚨 **상담원 연결 권고**" if result.escalate else "")
    )

    # 참조 출처
    sources_text = ""
    if result.sources:
        sources_text = "\n\n".join(f"• {src}" for src in result.sources)
    else:
        sources_text = "참조한 문서가 없습니다."

    return "", history, sentiment_text, sources_text


def reset_chat() -> tuple[list, str, str]:
    """대화 히스토리 초기화"""
    from services import chat_service as cs
    cs.clear_history(_session_id)
    return [], "", ""


# ── Gradio UI 조립 ────────────────────────────────────────────────────────
with gr.Blocks(title="SentiChat — 감정 분석 챗봇", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🤖 SentiChat\n"
        "**고객 문의 감정 분석 기반 AI 챗봇 서비스**\n\n"
        "고객의 감정 상태를 실시간으로 분석하고 RAG 지식베이스를 참조하여 응답합니다."
    )

    with gr.Row():
        with gr.Column(scale=2):
            chatbot  = gr.Chatbot(height=450, label="대화")
            msg_box  = gr.Textbox(
                placeholder="고객 문의를 입력하세요… (예: 배송이 너무 늦어요)",
                label="메시지",
                lines=2,
            )
            with gr.Row():
                send_btn  = gr.Button("전송 💬", variant="primary")
                reset_btn = gr.Button("초기화 🔄")

        with gr.Column(scale=1):
            with gr.Tab("📊 감정 분석"):
                sentiment_box = gr.Markdown(label="감정 결과", value="문의를 입력하면 감정이 분석됩니다.")
            with gr.Tab("📚 참조 출처"):
                sources_box = gr.Markdown(label="RAG 출처", value="FAQ / 정책 문서 참조 여부가 표시됩니다.")

    gr.Examples(
        examples=[
            ["배송이 2주째 안 와요. 너무 화가 납니다."],
            ["교환/환불은 어떻게 하나요?"],
            ["상품 잘 받았어요! 정말 마음에 들어요 😊"],
            ["주문한 것과 다른 색상이 왔는데 어떻게 해야 하나요?"],
        ],
        inputs=msg_box,
    )

    # ── 이벤트 바인딩 ─────────────────────────────────────────────────
    send_btn.click(
        respond,
        inputs=[msg_box, chatbot],
        outputs=[msg_box, chatbot, sentiment_box, sources_box],
    )
    msg_box.submit(
        respond,
        inputs=[msg_box, chatbot],
        outputs=[msg_box, chatbot, sentiment_box, sources_box],
    )
    reset_btn.click(reset_chat, outputs=[chatbot, sentiment_box, sources_box])


if __name__ == "__main__":
    demo.launch(
        server_port = cfg.gradio_port,
        share       = cfg.gradio_share,
        inbrowser   = True,
    )
