"""ChromaDB 向量库封装 —— 建库写入与持久化加载(对应主项目 rag/vector_store.py 的职责)。"""

from __future__ import annotations

import shutil

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma

from rag.config.chroma_config import COLLECTION_NAME, VECTOR_DIR
from rag.config.model_config import BATCH_SIZE
from rag.models import build_embeddings


def build_vector_store(chunks: list) -> Chroma:
    """分批向量化写入 ChromaDB;重建索引时清空旧目录,保证幂等(重复运行不会叠加重复切片)。"""
    if VECTOR_DIR.exists():
        print(f"⚠️ 已存在 {VECTOR_DIR.name}/,删除旧索引后重建 ...")
        shutil.rmtree(VECTOR_DIR)

    client = chromadb.PersistentClient(
        path=str(VECTOR_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    vs = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=build_embeddings(),
    )

    # ⚠️ 关键:按 BATCH_SIZE 条一批分批写入(该 embedding 模型的单请求上限)
    total = len(chunks)
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        vs.add_documents(batch)
        print(f"  已入库 {min(i + BATCH_SIZE, total)}/{total} 条切片")
    return vs


def get_vector_store() -> Chroma:
    """加载建库产出的持久化向量库(检索与建库使用同一 embedding 模型,保证语义空间一致)。"""
    if not VECTOR_DIR.exists():
        raise FileNotFoundError(f"未找到 {VECTOR_DIR},请先运行: python -m rag.ingest")

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
        raise RuntimeError("向量库为空,请先运行: python -m rag.ingest")
    return vs
