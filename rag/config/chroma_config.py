"""ChromaDB / 检索配置 —— 向量库路径、切片参数与检索质量参数。

对应主项目的 config/chroma.yaml 职责:向量库与检索相关参数集中在此。
"""

from __future__ import annotations

from rag.config import BASE_DIR

# ---------------------------------------------------------------- 向量库
VECTOR_DIR = BASE_DIR / "chromadb_data"      # ChromaDB 持久化目录
COLLECTION_NAME = "project_task_book"

# ---------------------------------------------------------------- 切片参数
CHUNK_SIZE = 500        # 中文按字符数计,500 字符 ≈ 一小段/半张表
CHUNK_OVERLAP = 80      # 片段间重叠,避免表格行被切断后丢失上下文

# ---------------------------------------------------------------- 检索质量参数
TOP_K = 5               # 检索工具召回条数
SEP = "\n\n---\n\n"     # 多片段之间的分隔符(judge 节点靠它拆开;evals 适配器同步使用)
MAX_REWRITES = 1        # 查询改写最多轮数,防止 judge↔rewrite 死循环

# ---------------------------------------------------------------- 混合检索参数(hybrid_search/ → agent_tool.hybrid_search_project_doc)
HYBRID_VECTOR_TOP_K = 10   # 向量召回条数(语义一路)
HYBRID_BM25_TOP_K = 10     # BM25 召回条数(关键词一路)
HYBRID_RERANK_TOP_K = 5    # BGE 重排后保留条数(最终喂给 LLM 的片段数)
