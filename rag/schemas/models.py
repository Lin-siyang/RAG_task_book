"""Pydantic 校验模型与 LangGraph State 定义。"""

from __future__ import annotations

from typing import Literal

from typing_extensions import Annotated, TypedDict

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class RelevanceGrade(BaseModel):
    """检索片段相关性评判结果。"""

    grade: Literal["yes", "no"] = Field(
        description="该片段是否包含有助于回答用户问题的信息:yes=有效,no=无效。只能填 yes 或 no。"
    )


class RagState(MessagesState):
    """所有对话消息都在 MessagesState.messages 里(自动累加);
    另加两个辅助字段:judge 选出的有效片段、已改写次数。"""

    relevant_chunks: Annotated[list[str], "judge 节点判为 yes 的片段"]
    rewrite_count: Annotated[int, "查询改写已执行的轮数"]
