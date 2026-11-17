"""BM25 稀疏检索器 —— jieba 中文分词 + rank-bm25,从向量库全量切片构建索引。

与 dense 检索的分工:BM25 擅长关键词精确匹配(编号、专有名词、表格条目),
dense 擅长语义相似;混合检索把两路召回合并后交给重排模型精挑。

索引缓存:首次检索时从向量库取全量切片建索引,之后复用;
并记住建索引时的 collection 条数 —— 条数变化(重建库/换文档)时自动重建。
"""

from __future__ import annotations

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# 索引缓存(进程内单例):_bm25 = BM25 索引,_docs = 与索引行对齐的全量切片
_vs = None
_bm25: BM25Okapi | None = None
_docs: list[Document] = []
_indexed_count: int | None = None


def _get_vs():
    """惰性获取向量库单例(函数内 import,避免与本包被 agent_tool 导入时形成加载顺序耦合)。"""
    global _vs
    if _vs is None:
        from rag.vector_store import get_vector_store

        _vs = get_vector_store()
    return _vs


def get_bm25_index() -> tuple[BM25Okapi, list[Document]]:
    """取 BM25 索引与全量切片;缓存失效条件 = collection 条数变化。"""
    global _bm25, _docs, _indexed_count
    vs = _get_vs()
    count = vs._collection.count()

    if _bm25 is None or _indexed_count != count:
        data = vs.get(include=["documents", "metadatas"])
        _docs = [
            Document(page_content=content, metadata=meta or {})
            for content, meta in zip(data["documents"], data["metadatas"])
        ]
        _bm25 = BM25Okapi([jieba.lcut(d.page_content) for d in _docs])
        _indexed_count = count
        print(f"  [bm25] 索引已构建: {count} 个切片")
    return _bm25, _docs


def bm25_search(query: str, k: int = 10) -> list[Document]:
    """BM25 检索:返回得分最高的 k 个切片(零分切片 = 无关键词重叠,过滤掉)。"""
    bm25, docs = get_bm25_index()
    scores = bm25.get_scores(jieba.lcut(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in ranked[:k] if scores[i] > 0]
