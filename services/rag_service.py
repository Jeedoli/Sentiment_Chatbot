"""
services/rag_service.py
────────────────────────
FAISS 벡터스토어 기반 지식 검색 서비스.

build_vectorstore.py 실행 후 생성된
data/vectorstore/ 인덱스를 로드하여 관련 문서를 검색합니다.
"""

import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import get_settings
from core.logging import logger


_vectorstore_cache: FAISS | None = None
_vectorstore_loaded: bool = False


def get_vectorstore() -> FAISS | None:
    """벡터스토어 singleton 로더.
    
    - 파일이 없으면 None 반환 (None은 캐싱하지 않아 이후 빌드 후 재시도 가능)
    - 성공적으로 로드된 경우에만 캐싱
    """
    global _vectorstore_cache, _vectorstore_loaded

    if _vectorstore_loaded:
        return _vectorstore_cache

    cfg  = get_settings()
    path = cfg.vectorstore_path

    if not os.path.exists(os.path.join(path, "index.faiss")):
        logger.warning(
            f"벡터스토어 없음: {path}/index.faiss\n"
            "poetry run python scripts/build_vectorstore.py 를 먼저 실행하세요."
        )
        return None  # None은 캐싱하지 않음 — 빌드 후 재호출 가능

    embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model)
    db         = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    logger.info(f"벡터스토어 로드 완료: {path}")
    _vectorstore_cache  = db
    _vectorstore_loaded = True
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
