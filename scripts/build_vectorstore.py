"""
scripts/build_vectorstore.py
─────────────────────────────
knowledge_base/ 폴더의 텍스트 파일을 읽어
FAISS 벡터스토어를 생성합니다.

[실행]
  poetry run python scripts/build_vectorstore.py

[입력]
  knowledge_base/*.txt   ← FAQ, 교환/환불정책, 배송안내 등 텍스트 파일

[출력]
  data/vectorstore/      ← FAISS 인덱스 파일 (index.faiss, index.pkl)
"""

import glob
import os

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


KB_DIR    = "knowledge_base"
VS_DIR    = "data/vectorstore"
CHUNK     = 400
OVERLAP   = 50


def load_documents() -> list[str]:
    docs   = []
    paths  = sorted(glob.glob(os.path.join(KB_DIR, "*.txt")))

    if not paths:
        raise FileNotFoundError(
            f"{KB_DIR}/ 에 .txt 파일이 없습니다.\n"
            "knowledge_base/faq.txt 같은 파일을 먼저 만들어주세요."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK, chunk_overlap=OVERLAP, separators=["\n\n", "\n", "。", " "]
    )

    for path in paths:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        chunks = splitter.split_text(raw)
        docs.extend(chunks)
        print(f"  {os.path.basename(path)}: {len(chunks)}개 청크")

    return docs


def build() -> None:
    print(f"[build_vectorstore] {KB_DIR}/ 텍스트 로드 중…")
    docs = load_documents()
    print(f"  총 {len(docs)}개 청크")

    print("[build_vectorstore] 임베딩 생성 중… (OpenAI API 호출)")
    embeddings = OpenAIEmbeddings()          # .env의 OPENAI_API_KEY 사용
    db         = FAISS.from_texts(docs, embeddings)

    os.makedirs(VS_DIR, exist_ok=True)
    db.save_local(VS_DIR)
    print(f"[build_vectorstore] 완료 → {VS_DIR}/")


if __name__ == "__main__":
    build()
