"""schemas 子包 —— Pydantic 模型类与 LangGraph State 定义。

    models.py  RelevanceGrade(judge 结构化输出) / RagState(图状态)

说明:放在 rag/ 下而非仓库根,是因为当前环境里 chromadb 依赖一个名为 schemas 的
第三方包(namespace package),顶层同名包会遮蔽它、导致向量库加载崩溃。
"""

from rag.schemas.models import RagState, RelevanceGrade

__all__ = ["RagState", "RelevanceGrade"]
