"""
scripts/build_dataset.py
─────────────────────────
AI Hub 라벨링 데이터 전체를 읽어 균형 잡힌 학습 데이터를 구성합니다.

[데이터 소스]
  data/aihub_dataset/Training/02.라벨링데이터/*.zip
    → 각 ZIP 안 JSON 파일의 'RawText', 'GeneralPolarity' 필드 사용
    → GeneralPolarity: -1=부정, 0=중립, 1=긍정  →  학습 라벨: 0, 1, 2

[전략]
  1. 전체 200K 中 부정/중립/긍정 모두 수집
  2. 클래스 불균형 보정: 부정/중립 전부 사용, 긍정은 같은 비율로 언더샘플링
     → 목표: 클래스별 최대 MAX_PER_CLASS 개 (기본 20,000)
  3. 합성 채팅 데이터(synthetic_chats.csv) 추가 → 챗 스타일 텍스트 커버
  4. 텍스트 정제 (URL 제거, 연속 공백/줄바꿈 정리, 빈 텍스트 필터링)
  5. Stratified 8:1:1 분할

[실행]
  poetry run python scripts/build_dataset.py
  poetry run python scripts/build_dataset.py --max_per_class 15000
"""

import argparse
import glob
import json
import os
import re
import zipfile
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split

OUT_DIR          = "data/processed"
LABEL_DIR        = "data/aihub_dataset/Training/02.라벨링데이터"
CHAT_CSV         = "data/raw/synthetic_chats.csv"
RAW_OUT          = "data/raw/aihub_all.csv"

# 클래스별 최대 샘플 수 (부정/중립이 적으므로 긍정을 이 값으로 제한)
DEFAULT_MAX_PER_CLASS = 20_000

# GeneralPolarity 매핑: AI Hub(-1,0,1) → 학습 라벨(0,1,2)
POL_MAP = {-1: 0, 0: 1, 1: 2}


# ── 텍스트 정제 ──────────────────────────────────────────────────────────
_URL_RE     = re.compile(r"https?://\S+|www\.\S+")
_TAG_RE     = re.compile(r"<[^>]+>")
_MULTI_SP   = re.compile(r"[ \t]+")
_MULTI_NL   = re.compile(r"\n{2,}")


def clean_text(text: str) -> str:
    """URL 제거 → HTML 태그 제거 → 연속 공백/줄바꿈 정리"""
    text = _URL_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = _MULTI_SP.sub(" ", text)
    text = _MULTI_NL.sub("\n", text)
    return text.strip()


# ── AI Hub 라벨 JSON 전체 읽기 ────────────────────────────────────────────
def load_aihub(label_dir: str) -> pd.DataFrame:
    """
    02.라벨링데이터/*.zip 안의 JSON 파일을 모두 읽어
    (text, label) 데이터프레임으로 반환.

    GeneralPolarity 값이 없거나 정수 변환 불가한 항목은 건너뜁니다.
    """
    buckets: dict[int, list[str]] = defaultdict(list)  # label → [text, ...]
    zip_paths = sorted(glob.glob(os.path.join(label_dir, "*.zip")))

    if not zip_paths:
        raise FileNotFoundError(
            f"라벨링 zip 파일이 없습니다: {label_dir}\n"
            "data/aihub_dataset/Training/02.라벨링데이터/ 경로를 확인하세요."
        )

    for zpath in zip_paths:
        zname = os.path.basename(zpath)
        count = 0
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():
                if not name.endswith(".json"):
                    continue
                with z.open(name) as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        continue
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        raw_pol = item.get("GeneralPolarity")
                        raw_txt = item.get("RawText", "")
                        if not raw_txt:
                            continue
                        try:
                            pol = int(raw_pol)
                        except (ValueError, TypeError):
                            continue
                        if pol not in POL_MAP:
                            continue
                        label = POL_MAP[pol]
                        txt = clean_text(str(raw_txt))
                        if len(txt) < 5:          # 너무 짧은 텍스트 제외
                            continue
                        buckets[label].append(txt)
                        count += 1
        print(f"  {zname}: {count:,}개 로드")

    # bucket → DataFrame
    rows = []
    for label, texts in buckets.items():
        for t in texts:
            rows.append({"text": t, "label": label})

    df = pd.DataFrame(rows)
    print(f"\n[build_dataset] AI Hub 원본 합계: {len(df):,}개")
    for lbl, cnt in df["label"].value_counts().sort_index().items():
        name = {0: "부정", 1: "중립", 2: "긍정"}[lbl]
        print(f"  {name}({lbl}): {cnt:,}개")
    return df


# ── 클래스 균형 맞추기 ────────────────────────────────────────────────────
def balance(df: pd.DataFrame, max_per_class: int, seed: int = 42) -> pd.DataFrame:
    """
    각 클래스별로 max_per_class 개 이하로 제한(undersampling).
    부족한 클래스는 그대로 전부 사용하여 정보 손실 최소화.
    """
    parts = []
    for lbl in sorted(df["label"].unique()):
        sub = df[df["label"] == lbl]
        if len(sub) > max_per_class:
            sub = sub.sample(max_per_class, random_state=seed)
        parts.append(sub)
    balanced = pd.concat(parts, ignore_index=True)
    return balanced.sample(frac=1, random_state=seed).reset_index(drop=True)


# ── 합성 채팅 데이터 추가 ────────────────────────────────────────────────
def add_chat_data(df: pd.DataFrame, chat_csv: str) -> pd.DataFrame:
    """
    synthetic_chats.csv (챗봇 스타일 텍스트)를 학습 데이터에 합산.
    실제 서비스가 받는 채팅 스타일 입력에 모델이 취약해지지 않도록 추가.
    """
    if not os.path.exists(chat_csv):
        print(f"[build_dataset] 합성 채팅 CSV 없음, 건너뜀: {chat_csv}")
        return df

    chat_df = pd.read_csv(chat_csv)
    chat_df = chat_df[["text", "label"]].dropna()
    chat_df["label"] = chat_df["label"].astype(int)
    print(f"[build_dataset] 합성 채팅 데이터 {len(chat_df):,}개 추가")
    return pd.concat([df, chat_df], ignore_index=True)


# ── 분할 후 저장 ─────────────────────────────────────────────────────────
def save_splits(df: pd.DataFrame) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    df_tr, df_tmp = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    df_val, df_te = train_test_split(
        df_tmp, test_size=0.5, stratify=df_tmp["label"], random_state=42
    )

    df_tr.to_csv(f"{OUT_DIR}/train.csv",  index=False, encoding="utf-8-sig")
    df_val.to_csv(f"{OUT_DIR}/val.csv",   index=False, encoding="utf-8-sig")
    df_te.to_csv(f"{OUT_DIR}/test.csv",   index=False, encoding="utf-8-sig")

    print(f"\n[build_dataset] 저장 완료")
    print(f"  train : {len(df_tr):,}개")
    print(f"  val   : {len(df_val):,}개")
    print(f"  test  : {len(df_te):,}개")
    for split, path in [("train", f"{OUT_DIR}/train.csv"), ("val", f"{OUT_DIR}/val.csv")]:
        tmp = pd.read_csv(path)
        dist = tmp["label"].value_counts().sort_index()
        label_map = {0: "부정", 1: "중립", 2: "긍정"}
        print(f"  {split} 분포: { {label_map[k]: v for k, v in dist.items()} }")


# ── 메인 ─────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="AI Hub → 균형 데이터셋 구축")
    parser.add_argument(
        "--max_per_class", type=int, default=DEFAULT_MAX_PER_CLASS,
        help=f"클래스별 최대 샘플 수 (기본: {DEFAULT_MAX_PER_CLASS})"
    )
    parser.add_argument(
        "--label_dir", default=LABEL_DIR,
        help="AI Hub 라벨링 zip 디렉터리"
    )
    parser.add_argument(
        "--chat_csv", default=CHAT_CSV,
        help="합성 채팅 데이터 CSV 경로"
    )
    args = parser.parse_args()

    print("[build_dataset] AI Hub 라벨링 데이터 로드 시작…")
    df = load_aihub(args.label_dir)

    print(f"\n[build_dataset] 클래스 균형 조정 (max_per_class={args.max_per_class:,})…")
    df = balance(df, args.max_per_class)

    print(f"\n[build_dataset] 균형 후:")
    for lbl, cnt in df["label"].value_counts().sort_index().items():
        name = {0: "부정", 1: "중립", 2: "긍정"}[lbl]
        print(f"  {name}({lbl}): {cnt:,}개")

    df = add_chat_data(df, args.chat_csv)

    print(f"\n[build_dataset] 최종 전체: {len(df):,}개")
    save_splits(df)


if __name__ == "__main__":
    main()
