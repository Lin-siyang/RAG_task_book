"""模型加载子包 —— 统一从 factory 获取 Chat / Embedding 模型实例。"""

from rag.models.factory import build_embeddings, make_llm

__all__ = ["build_embeddings", "make_llm"]
