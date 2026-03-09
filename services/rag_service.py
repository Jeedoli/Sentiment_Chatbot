"""
services/rag_service.py
────────────────────────
FAISS 벡터스토어 기반 지식 검색 서비스.

build_vectorstore.py 실행 후 생성된
data/vectorstore/ 인덱스를 로드하여 관련 문서를 검색합니다.
"""

import os
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from core.config import get_settings
from core.logging import logger


@lru_cache
def get_vectorstore() -> FAISS | None:
    """벡터스토어 singleton 로더. 파일이 없으면 None 반환."""
    cfg  = get_settings()
    path = cfg.vectorstore_path

    if not os.path.exists(os.path.join(path, "index.faiss")):
        logger.warning(
            f"벡터스토어 없음: {path}/index.faiss\n"
            "poetry run python scripts/build_vectorstore.py 를 먼저 실행하세요."
        )
        return None

    embeddings = OpenAIEmbeddings(model=cfg.embedding_model)
    db         = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    logger.info(f"벡터스토어 로드 완료: {path}")
    return db


def retrieve(query: str) -> list[str]:
    """
    쿼리와 가장 관련 있는 k개의 문서 청크를 반환.
    벡터스토어가 없으면 빈 리스트 반환.
    """
    cfg = get_settings()
    db  = get_vectorstore()
    if db is None:
        return []

    docs = db.similarity_search(query, k=cfg.top_k_retrieval)
    return [d.page_content for d in docs]
