"""ChromaDB 向量库加载 —— 读取 ingest 阶段构建的持久化索引。"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma

from rag.config import COLLECTION_NAME, VECTOR_DIR
from rag.models import build_embeddings


def get_vector_store() -> Chroma:
    """加载建库脚本产出的持久化向量库(检索与建库使用同一 embedding 模型,保证语义空间一致)。"""
    if not VECTOR_DIR.exists():
        raise FileNotFoundError(f"未找到 {VECTOR_DIR},请先运行: python ingest/ingest_pdf.py")

    client = chromadb.PersistentClient(
        path=str(VECTOR_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    vs = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=build_embeddings(),
    )
    if vs._collection.count() == 0:
        raise RuntimeError("向量库为空,请先运行: python ingest/ingest_pdf.py")
    return vs
