"""配置子包 —— 按主题拆分为三个配置文件(对应主项目 backend/app/config/ 的组织方式):

    model_config   模型配置(Chat / Embedding,值从 .env 读取)
    prompt_config  提示词配置(模板目录 + 加载器,读取 prompts/*.txt)
    chroma_config  ChromaDB / 检索配置(向量库路径、切片、召回参数)

本模块(__init__)只保留:控制台编码引导、.env 加载、跨配置共享的路径常量。
改配置只动对应主题文件(或 .env),不改业务代码。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Windows 控制台 UTF-8 输出(中文/emoji 打印不乱码;导入本包即生效,与拆分前行为一致)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

# 从本目录向上查找 .env(兼容从任意工作目录/子目录启动;python-dotenv 默认不覆盖已有变量,重复加载无副作用)
for _parent in Path(__file__).resolve().parents:
    if (_parent / ".env").exists():
        load_dotenv(_parent / ".env")
        break
else:
    load_dotenv()

# LangSmith 追踪(setdefault:.env 中的配置优先生效)
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "RAGas_task_book")

# ---------------------------------------------------------------- 共享路径
BASE_DIR = Path(__file__).resolve().parent.parent.parent   # config/ → rag/ → RAG_task_book/ 仓库根
RESOURCES_DIR = BASE_DIR / "resources"
PDF_PATH = RESOURCES_DIR / "重点新材料研发及应用国家科技重大专项课题任务书.pdf"
