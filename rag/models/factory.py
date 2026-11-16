"""模型工厂 —— Chat / Embedding 模型的统一构建入口(对应主项目 utils/factory.py)。

模型配置(config/model_config.py)从 .env 读取,本文件只负责把配置组装成模型实例。
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag.config.model_config import BATCH_SIZE, chat_settings, embedding_settings


def build_embeddings() -> OpenAIEmbeddings:
    """qwen3.7-text-embedding-flash(OpenAI 兼容接口,建库与检索共用同一实例配置)。

    check_embedding_ctx_length=False:不走 OpenAI 的 tiktoken 预切分;
    chunk_size=BATCH_SIZE:langchain 内部每次 API 请求最多发 20 条(模型单请求上限)。
    """
    return OpenAIEmbeddings(
        **embedding_settings(),
        check_embedding_ctx_length=False,
        chunk_size=BATCH_SIZE,
    )


def make_llm(temperature: float = 0) -> ChatOpenAI:
    """Chat 模型走 .env 里的 DeepSeek(已验证可用);想换模型改 .env 或本文件。"""
    return ChatOpenAI(
        **chat_settings(),
        temperature=temperature,
    )
