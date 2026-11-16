"""API 服务入口 —— FastAPI 应用装配(对应主项目 backend/main.py 的职责)。

职责:
    1. 创建 FastAPI 应用,注册聊天路由
    2. 配置 CORS:允许访问的前端地址从环境变量 API_ALLOW_ORIGINS 读取(逗号分隔白名单)
    3. 服务启动时(lifespan)加载知识库、准备 Agent(chat_service.warmup)

跑法:
    uvicorn api.main:app --reload --port 8100     # 在仓库根目录运行(8100 避免与主项目 8000 冲突)

接口一览:
    GET  /health               确认服务进程运行
    POST /api/v1/chat          提交问题,一次性返回完整回答
    POST /api/v1/chat/stream   提交问题,SSE 逐段返回回答
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# api/ 不在仓库根,直跑时把仓库根加入 sys.path(agent/ rag/ 等包才能导入)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

# 与 rag/config 一致:从本目录向上查找 .env(CORS 白名单在模块级读取,须先于 lifespan 加载)
for _parent in Path(__file__).resolve().parents:
    if (_parent / ".env").exists():
        load_dotenv(_parent / ".env")
        break
else:
    load_dotenv()

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from api.routers import chat_router
from api.services import chat_service

# CORS 白名单:逗号分隔的前端地址;浏览器凭据模式下禁止 "*",必须显式列出
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("API_ALLOW_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    if origin.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动:加载知识库、准备 Agent(失败则服务起不来,快速暴露问题);关闭:无需清理(均为进程内资源)。"""
    chat_service.warmup()
    print("✅ 知识库与 Agent 已就绪,可以接收请求")
    yield


app = FastAPI(title="RAG_task_book API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat_router)
