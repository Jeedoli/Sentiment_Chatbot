"""
scripts/preprocess.py
──────────────────────
다양한 소스의 텍스트 감정 데이터를 정제하여
학습 가능한 CSV 형태로 저장.

[지원 데이터소스]
  A. HuggingFace datasets (nsmc / kornli 등)
  B. 로컬 CSV 파일 (path 지정)
  C. 합성 샘플 데이터 (데이터 없을 때 데모용)

[실행]
  poetry run python scripts/preprocess.py --source nsmc
  poetry run python scripts/preprocess.py --source local --csv_path data/raw/reviews.csv
  poetry run python scripts/preprocess.py --source sample

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


# ── C. 합성 샘플 데이터 ──────────────────────────────────────────────────
def make_sample_data() -> pd.DataFrame:
    """
    데이터가 없을 때 파이프라인 동작을 확인하기 위한 데모용 샘플.
    실제 서비스에서는 사용하지 마세요.
    """
    rows = [
        # 부정(0)
        ("배송이 너무 늦어요. 3주째 기다리고 있어요.", 0),
        ("상품이 파손된 채로 도착했습니다. 환불 요청합니다.", 0),
        ("설명과 전혀 다른 제품이에요. 실망이에요.", 0),
        ("고객센터 연결이 안 됩니다. 화가 많이 납니다.", 0),
        ("이런 제품 처음 봤어요. 진짜 불량품이에요.", 0),
        ("사기당한 기분이에요. 다신 여기서 안 삽니다.", 0),
        ("포장이 엉망이고 제품도 이미 뜯겨있었어요.", 0),
        ("AS 거부당했어요. 너무 부당한 처우예요.", 0),
        # 중립(1)
        ("제품은 평범하고 배송은 보통이었습니다.", 1),
        ("나쁘지 않은데 특별히 좋지도 않아요.", 1),
        ("가격 대비 무난한 상품입니다.", 1),
        ("배송 조금 늦었지만 제품은 괜찮았어요.", 1),
        ("기대한 것과 비슷하게 도착했습니다.", 1),
        ("좋은 점도 있고 아쉬운 점도 있어요.", 1),
        # 긍정(2)
        ("포장이 꼼꼼하고 빠른 배송 감사해요!", 2),
        ("상품 품질이 정말 좋아요. 재구매 의사 있어요.", 2),
        ("친절한 고객서비스 덕분에 문제가 금방 해결됐어요.", 2),
        ("가격도 착하고 품질도 최고예요!", 2),
        ("설명대로 딱 맞게 와서 너무 만족스러워요.", 2),
        ("완벽한 구매였어요. 주변에 추천하겠습니다!", 2),
    ]
    return pd.DataFrame(rows, columns=["text", "label"])


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
    parser.add_argument("--source",   default="sample",
                        choices=["nsmc", "local", "sample"],
                        help="데이터 소스 선택")
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
        print("[preprocess] 샘플 데이터 생성 중…")
        df = make_sample_data()

    print(f"[preprocess] 원본 데이터 {len(df):,} rows")
    save_splits(df)


if __name__ == "__main__":
    main()
