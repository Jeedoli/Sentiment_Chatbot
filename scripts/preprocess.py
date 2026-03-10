"""
scripts/preprocess.py
──────────────────────
다양한 소스의 텍스트 감정 데이터를 정제하여
학습 가능한 CSV 형태로 저장.

[지원 데이터소스]
  A. HuggingFace datasets (nsmc / kornli 등)
  B. 로컬 CSV 파일 (path 지정)
     → 먼저 scripts/generate_samples.py 를 실행해 data/raw/synthetic_reviews.csv 생성

[실행]
  poetry run python scripts/preprocess.py --source nsmc
  poetry run python scripts/preprocess.py --source local --csv_path data/raw/synthetic_reviews.csv

[출력]
  data/processed/train.csv
  data/processed/val.csv
  data/processed/test.csv

  컬럼: text, label   (0=부정, 1=중립, 2=긍정)
"""

import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


OUT_DIR = "data/processed"


# ── A. HuggingFace NSMC → 3-class 변환 ────────────────────────────────────
def load_nsmc() -> pd.DataFrame:
    """
    네이버 영화리뷰 감정 데이터(NSMC).
    원본 라벨: 0=부정, 1=긍정 → 여기서는 중립 없이 2-class.
    ∴ 긍정(1)→2, 부정(0)→0 으로 맞춰서 중립(1)은 비어있음.
    """
    from datasets import load_dataset
    ds = load_dataset("nsmc")
    df_train = ds["train"].to_pandas()
    df_test  = ds["test"].to_pandas()
    df       = pd.concat([df_train, df_test], ignore_index=True)

    # HuggingFace nsmc 컬럼: 'document', 'label'  (0=neg, 1=pos)
    df = df.dropna(subset=["document", "label"])
    df = df.rename(columns={"document": "text"})
    # 2-class → 3-class 매핑 (0=neg, 2=pos)
    df["label"] = df["label"].map({0: 0, 1: 2})
    return df[["text", "label"]]


# ── B. 로컬 CSV 로드 ─────────────────────────────────────────────────────
def load_local_csv(path: str) -> pd.DataFrame:
    """
    로컬 CSV를 불러옵니다.

    CSV 형식 가이드:
      - text    : 텍스트 컬럼 (필수)
      - label   : 0/1/2 정수 또는 '부정'/'중립'/'긍정' 문자열 (필수)
      - rating  : 별점(1-5) 있으면 자동 label 변환
    """
    df = pd.read_csv(path)

    # 별점으로 라벨 자동 생성 옵션
    if "rating" in df.columns and "label" not in df.columns:
        df["label"] = df["rating"].apply(lambda r: 0 if r <= 2 else (1 if r == 3 else 2))

    # 한글 레이블 → 정수 변환
    if df["label"].dtype == object:
        kmap = {"부정": 0, "중립": 1, "긍정": 2, "negative": 0, "neutral": 1, "positive": 2}
        df["label"] = df["label"].str.strip().map(kmap)

    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    return df[["text", "label"]]


# ── 분할 후 저장 ─────────────────────────────────────────────────────────
def save_splits(df: pd.DataFrame) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # 스트래티파이드 8:1:1 분할
    df_tr, df_tmp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    df_val, df_te = train_test_split(df_tmp, test_size=0.5, stratify=df_tmp["label"], random_state=42)

    df_tr.to_csv(f"{OUT_DIR}/train.csv", index=False, encoding="utf-8-sig")
    df_val.to_csv(f"{OUT_DIR}/val.csv",  index=False, encoding="utf-8-sig")
    df_te.to_csv(f"{OUT_DIR}/test.csv",  index=False, encoding="utf-8-sig")

    print(f"[preprocess] 완료")
    print(f"  train: {len(df_tr):,} rows")
    print(f"  val  : {len(df_val):,} rows")
    print(f"  test : {len(df_te):,} rows")
    for split, path in [("train", f"{OUT_DIR}/train.csv"), ("val", f"{OUT_DIR}/val.csv")]:
        tmp = pd.read_csv(path)
        print(f"  {split} 라벨 분포: {dict(tmp['label'].value_counts().sort_index())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="감정 데이터 전처리")
    parser.add_argument("--source",   default="local",
                        choices=["nsmc", "local"],
                        help='데이터 소스 선택 ("nsmc" or "local")')
    parser.add_argument("--csv_path", default="data/raw/reviews.csv",
                        help="--source local 사용 시 CSV 경로")
    args = parser.parse_args()

    if args.source == "nsmc":
        print("[preprocess] NSMC 데이터셋 로드 중…")
        df = load_nsmc()
    elif args.source == "local":
        print(f"[preprocess] 로컬 CSV 로드: {args.csv_path}")
        df = load_local_csv(args.csv_path)
    else:
        raise ValueError(f"지원하지 않는 소스: {args.source}")

    print(f"[preprocess] 원본 데이터 {len(df):,} rows")
    save_splits(df)


if __name__ == "__main__":
    main()
