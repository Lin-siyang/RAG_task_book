"""聊天路由 —— 定义 API 网址与请求方式(对应主项目 backend/app/router/chat.py 的职责)。

    GET  /health               确认服务进程运行
    POST /api/v1/chat          提交问题,一次性返回完整回答(便于先调通)
    POST /api/v1/chat/stream   提交问题,逐段返回回答(供聊天页面使用)
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from api.schemas.chat import ChatRequest, ChatResponse
from api.services import chat_service

chat_router = APIRouter()


@chat_router.get("/health")
def health() -> dict:
    """存活检查:能响应即说明服务进程运行。"""
    return {"status": "ok"}


@chat_router.post("/api/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """一次性问答:接收问题,调用 chat_service 取得完整回答。"""
    return ChatResponse(answer=chat_service.chat(req.question))


@chat_router.post("/api/v1/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """流式问答:SSE 逐段返回回答片段,前端按 data: 行解析(事件协议见 chat_service)。"""
    return StreamingResponse(
        chat_service.chat_stream(req.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
