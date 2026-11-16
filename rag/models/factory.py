"""模型工厂 —— Chat / Embedding 模型的统一构建入口(对应主项目 utils/factory.py)。

模型配置全部来自 .env(EMBEDDING_* / DEEPSEEK_*),换模型只改 .env 或本文件。
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag.config import BATCH_SIZE


def build_embeddings() -> OpenAIEmbeddings:
    """qwen3.7-text-embedding-flash(OpenAI 兼容接口,建库与检索共用同一实例配置)。

    check_embedding_ctx_length=False:不走 OpenAI 的 tiktoken 预切分;
    chunk_size=BATCH_SIZE:langchain 内部每次 API 请求最多发 20 条(模型单请求上限)。
    """
    return OpenAIEmbeddings(
        model=os.environ["EMBEDDING_MODEL"],
        api_key=os.environ["EMBEDDING_API_KEY"],
        base_url=os.environ["EMBEDDING_URL"],
        check_embedding_ctx_length=False,
        chunk_size=BATCH_SIZE,
    )


def make_llm(temperature: float = 0) -> ChatOpenAI:
    """Chat 模型走 .env 里的 DeepSeek(已验证可用);想换模型改这里即可。"""
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ["DEEPSEEK_BASE_URL"],
        temperature=temperature,
    )
