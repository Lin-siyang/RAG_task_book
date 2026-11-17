"""BGE 重排器 —— bge-reranker-v2-m3(cross-encoder),对混合召回的候选切片按相关性精排。

模型约 2GB,进程内单例:首次调用加载(当前 torch 为 CPU 版,加载与推理稍慢),
后续复用。路径来自 .env 的 BGE_RERANKER_PATH(见 rag/config/model_config.reranker_path)。
"""

from __future__ import annotations

from rag.config.model_config import reranker_path

_reranker = None  # CrossEncoder 进程内单例


def get_reranker():
    """惰性加载重排模型(首次调用加载,后续复用)。"""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(reranker_path(), max_length=512)
        print("  [rerank] BGE-reranker-v2-m3 已加载")
    return _reranker


def rerank(query: str, docs: list, top_k: int = 5) -> list:
    """按 query-切片相关性分数降序重排,返回前 top_k 个 Document。

    重排分数写入 metadata["rerank_score"],便于调参与诊断。
    """
    if not docs:
        return []
    model = get_reranker()
    scores = model.predict([(query, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)[:top_k]
    for d, s in ranked:
        d.metadata["rerank_score"] = round(float(s), 4)
    return [d for d, _ in ranked]
