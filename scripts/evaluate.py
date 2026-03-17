"""학습된 모델 성능 검증 및 실전 예문 테스트"""
import torch
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score
from models.sentiment import SentimentInference

CKPT = "saved_models/sentiment_best.pt"
TEST_CSV = "data/processed/test.csv"

print("=== 모델 로드 중 ===")
infer = SentimentInference(ckpt_path=CKPT, model_name="klue/roberta-base", max_len=128)

# ── 테스트셋 평가
print("\n=== test.csv 평가 ===")
df = pd.read_csv(TEST_CSV)
preds = []
for text in df["text"].tolist():
    out = infer.predict(str(text))
    preds.append(out.label)

labels = df["label"].tolist()
f1 = f1_score(labels, preds, average="macro")
acc = accuracy_score(labels, preds)
print(f"  test F1-macro : {f1:.4f}")
print(f"  test Accuracy : {acc:.4f}")
print("\n" + classification_report(labels, preds, target_names=["부정", "중립", "긍정"], digits=4))

# ── 실전 예문 테스트
SAMPLES = [
    ("배송이 너무 늦어요. 정말 화나네요!", 0),
    ("개짜증나네…배송도 늦고 불량품이고", 0),
    ("ㅋㅋ 이게 뭐야 진짜 쓰레기네", 0),
    ("그저 그래요. 뭐 평범합니다.", 1),
    ("배송 언제 오나요? 확인 부탁드립니다.", 1),
    ("환불 방법 알려주세요", 1),
    ("진짜 너무 좋아서 주변에 다 추천했어요!", 2),
    ("역대급으로 만족스럽다 ㅠㅠ 최고!", 2),
    ("빠른 배송 감사합니다 :)", 2),
    ("ㄹㅇ 퀄리티 미쳤다 대박", 2),
]

label_map = {0: "부정", 1: "중립", 2: "긍정"}
print("=== 실전 예문 테스트 ===")
correct = 0
for text, expected in SAMPLES:
    out = infer.predict(text)
    emoji = "✅" if out.label == expected else "❌"
    correct += (out.label == expected)
    print(f"{emoji} [{label_map[expected]}→{out.label_str}] neg={out.negative:.2f} neu={out.neutral:.2f} pos={out.positive:.2f}  '{text[:40]}'")

print(f"\n실전 예문 정확도: {correct}/{len(SAMPLES)}")
