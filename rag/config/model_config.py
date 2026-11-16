"""模型配置 —— Chat / Embedding 模型的全部 .env 读取入口。

模型相关的环境变量只在这里出现,业务代码(models/factory.py)从这里取配置,
不在各处散落 os.getenv / os.environ。换模型只改 .env 或本文件。
"""

from __future__ import annotations

import os

# ⚠️ embedding 模型(qwen3.7-text-embedding-flash)单次请求上限 20 条:
# 建库分批写入(rag/ingest.py)与 langchain 内部 chunk_size(models/factory.py)都以此为准
BATCH_SIZE = 20


def embedding_settings() -> dict:
    """Embedding 模型配置(OpenAI 兼容接口,来自 .env 的 EMBEDDING_*)。"""
    return {
        "model": os.environ["EMBEDDING_MODEL"],
        "api_key": os.environ["EMBEDDING_API_KEY"],
        "base_url": os.environ["EMBEDDING_URL"],
    }


def chat_settings() -> dict:
    """Chat 模型配置(DeepSeek 官方 API,来自 .env 的 DEEPSEEK_*)。"""
    return {
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        "api_key": os.environ["DEEPSEEK_API_KEY"],
        "base_url": os.environ["DEEPSEEK_BASE_URL"],
    }
