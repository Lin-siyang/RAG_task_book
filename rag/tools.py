"""Agent 检索工具 —— 供 LangGraph ToolNode / agent 节点绑定的任务书检索工具。"""

from __future__ import annotations

from langchain.tools import tool

from rag.config import SEP, TOP_K
from rag.vector_store import get_vector_store

# 模块级单例:导入本模块即连库(与重构前 rag_agent.py 的行为一致,evals 依赖该加载副作用)
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
    用提炼后的关键词调用本工具;每次返回与查询最相关的 3 个片段(含页码)。

    Args:
        query: 检索查询词,应为中文关键词或短句,例如 "考核指标" "课题参加单位"。
    """
    docs = VECTOR_STORE.similarity_search(query, k=TOP_K)
    return format_docs(docs)
