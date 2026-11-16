"""聊天接口的数据格式 —— FastAPI 据此自动校验请求体、生成 /docs 接口文档。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """POST /api/v1/chat 与 POST /api/v1/chat/stream 的请求体。"""

    question: str = Field(min_length=1, description="用户问题,不能为空字符串")


class ChatResponse(BaseModel):
    """POST /api/v1/chat 的响应体(流式接口走 SSE 事件,不用此模型)。"""

    answer: str = Field(description="Agent 的完整回答")
