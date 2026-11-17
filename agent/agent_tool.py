"""Agent 检索工具 —— 供 LangGraph ToolNode / agent 节点绑定的任务书检索工具。

两个工具(在 agent/agent.py 里手动切换绑定):
    search_project_doc            纯向量语义检索(单路)
    hybrid_search_project_doc     向量 + BM25 双路召回 → BGE 重排(混合检索)
"""

from __future__ import annotations

from langchain.tools import tool

from rag.config.chroma_config import (
    HYBRID_BM25_TOP_K,
    HYBRID_RERANK_TOP_K,
    HYBRID_VECTOR_TOP_K,
    SEP,
    TOP_K,
)
from rag.hybrid_search.bm25_retriever import bm25_search
from rag.hybrid_search.reranker import rerank
from rag.vector_store import get_vector_store

# 模块级单例:导入本模块即连库(与重构前行为一致,evals 依赖该加载副作用)
VECTOR_STORE = get_vector_store()


def format_docs(docs: list) -> str:
    """把检索到的 Document 格式化为带页码的片段文本,片段间用 SEP 分隔。"""
    parts = []
    for i, d in enumerate(docs, 1):
        page = d.metadata.get("page", "?")
        parts.append(f"【片段{i} | 第{page}页】\n{d.page_content}")
    return SEP.join(parts)


@tool
def search_project_doc(query: str) -> str:
    """检索项目任务书《重点新材料研发及应用国家科技重大专项课题任务书》的内容片段。

    当用户询问课题的研究内容、考核指标、研究进度、经费、参加单位等任务书相关信息时,
    用提炼后的关键词调用本工具;每次返回与查询最相关的 5 个片段(含页码)。

    Args:
        query: 检索查询词,应为中文关键词或短句,例如 "考核指标" "课题参加单位"。
    """
    docs = VECTOR_STORE.similarity_search(query, k=TOP_K)
    return format_docs(docs)


@tool
def hybrid_search_project_doc(query: str) -> str:
    """混合检索项目任务书:向量语义检索 + BM25 关键词检索双路召回,经 BGE 重排模型精排后返回片段。

    与 search_project_doc 的区别:本工具先扩大召回(向量 10 条 + BM25 10 条)再由重排模型
    挑出最相关的 5 条,对课题编号、专有名词、表格条目等需要精确匹配的问题召回更稳。

    当用户询问课题的研究内容、考核指标、研究进度、经费、参加单位等任务书相关信息时,
    用提炼后的关键词调用本工具;每次返回重排后最相关的 5 个片段(含页码)。

    Args:
        query: 检索查询词,应为中文关键词或短句,例如 "考核指标" "课题参加单位"。
    """
    # 1. 双路召回:向量(语义相似) + BM25(关键词精确匹配)
    vector_docs = VECTOR_STORE.similarity_search(query, k=HYBRID_VECTOR_TOP_K)
    bm25_docs = bm25_search(query, k=HYBRID_BM25_TOP_K)

    # 2. 合并去重(同一切片可能被两路同时召回,按切片正文判重)
    merged, seen = [], set()
    for d in vector_docs + bm25_docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            merged.append(d)

    # 3. BGE 重排,保留最相关的 HYBRID_RERANK_TOP_K 条(分数写入 metadata["rerank_score"])
    top_docs = rerank(query, merged, top_k=HYBRID_RERANK_TOP_K)
    return format_docs(top_docs)
