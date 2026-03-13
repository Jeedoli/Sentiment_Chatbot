"""
scripts/train.py
─────────────────
SentimentClassifier (klue/roberta-base) 학습 스크립트.

[실행 전 준비]
  1. poetry run python scripts/preprocess.py --source nsmc   (or sample)
  2. poetry run pip install torch

[학습 실행]
  poetry run python scripts/train.py
  poetry run python scripts/train.py --epochs 10 --batch_size 16 --lr 3e-5

[출력]
  saved_models/sentiment_best.pt    ← val F1 기준 최적 체크포인트
  saved_models/sentiment_last.pt    ← 마지막 에폭 체크포인트
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

from models.sentiment import SentimentClassifier


# ── 데이터셋 클래스 ────────────────────────────────────────────────────────
class SentimentDataset(Dataset):
    """
    CSV 기반 Dataset.
    컬럼: text, label  (label: 0=부정, 1=중립, 2=긍정)
    """

    def __init__(self, csv_path: str, tokenizer, max_len: int = 128):
        self.df        = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row  = self.df.iloc[idx]
        enc  = self.tokenizer(
            str(row["text"]),
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),       # (seq_len,)
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(int(row["label"]), dtype=torch.long),
        }


# ── 학습 루프 ─────────────────────────────────────────────────────────────
def train(
    model_name:     str,
    train_csv:      str,
    val_csv:        str,
    epochs:         int,
    batch_size:     int,
    lr:             float,
    max_len:        int,
    warmup_ratio:   float,
    save_dir:       str,
    num_labels:     int = 3,
) -> None:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device} | model={model_name} | epochs={epochs}")

    # ── 토크나이저 ─────────────────────────────────────────────────────
    tokenizer     = AutoTokenizer.from_pretrained(model_name)

    # ── 데이터로더 ─────────────────────────────────────────────────────
    train_ds      = SentimentDataset(train_csv, tokenizer, max_len)
    val_ds        = SentimentDataset(val_csv,   tokenizer, max_len)
    train_loader  = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader    = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}")

    # ── 모델 / 옵티마이저 ──────────────────────────────────────────────
    model         = SentimentClassifier(model_name=model_name, num_labels=num_labels).to(device)
    optimizer     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps   = len(train_loader) * epochs
    warmup_steps  = int(total_steps * warmup_ratio)
    scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion     = nn.CrossEntropyLoss()

    os.makedirs(save_dir, exist_ok=True)
    best_f1       = 0.0

    # TensorBoard logging
    log_dir = os.path.join("runs", os.path.basename(save_dir))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[train] TensorBoard logs -> {log_dir}")

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        tot_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)

        for batch in pbar:
            input_ids  = batch["input_ids"].to(device)
            attn_mask  = batch["attention_mask"].to(device)
            labels     = batch["label"].to(device)

            logits     = model(input_ids, attn_mask)
            loss       = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            tot_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = tot_loss / len(train_loader)

        # ── Validation ─────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits  = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                preds   = logits.argmax(dim=-1).cpu().tolist()
                all_preds  += preds
                all_labels += batch["label"].tolist()

        f1 = f1_score(all_labels, all_preds, average="macro")
        val_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={avg_train:.4f}  val_f1={f1:.4f}  "
            f"lr={val_lr:.2e}"
        )

        # TensorBoard
        writer.add_scalar("train/loss", avg_train, epoch)
        writer.add_scalar("val/f1", f1, epoch)
        writer.add_scalar("val/lr", val_lr, epoch)

        # ── 베스트 저장 ─────────────────────────────────────────────────
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(save_dir, "sentiment_best.pt"))
            print(f"  ✅ Best model saved (val_f1={best_f1:.4f})")

    # 마지막 에폭 저장
    torch.save(model.state_dict(), os.path.join(save_dir, "sentiment_last.pt"))
    print(f"\n[train] 완료. Best val_F1={best_f1:.4f}")

    writer.flush()
    writer.close()

    # ── 최종 분류 리포트 ───────────────────────────────────────────────
    print("\n=== Classification Report ===")
    try:
        print(classification_report(all_labels, all_preds,
                                     target_names=["부정", "중립", "긍정"],
                                     digits=4))
    except ValueError:
        print("[train] classification report skipped (insufficient classes)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="감정 분류 모델 학습")
    p.add_argument("--model_name",    default="klue/roberta-base")
    p.add_argument("--train_csv",     default="data/processed/train.csv")
    p.add_argument("--val_csv",       default="data/processed/val.csv")
    p.add_argument("--epochs",        type=int,   default=5)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=2e-5)
    p.add_argument("--max_len",       type=int,   default=128)
    p.add_argument("--warmup_ratio",  type=float, default=0.1)
    p.add_argument("--save_dir",      default="saved_models")
    p.add_argument("--num_labels",    type=int,   default=3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_name   = args.model_name,
        train_csv    = args.train_csv,
        val_csv      = args.val_csv,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        max_len      = args.max_len,
        warmup_ratio = args.warmup_ratio,
        save_dir     = args.save_dir,
        num_labels   = args.num_labels,
    )
