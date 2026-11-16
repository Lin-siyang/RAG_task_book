"""集中配置 —— .env 加载、路径常量、RAG 参数(单一配置源,所有模块从这里取值)。

改模型/改参数只动这里(或 .env),不改动业务代码。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Windows 控制台 UTF-8 输出(中文/emoji 打印不乱码;导入本包即生效,与原入口脚本行为一致)
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

# ---------------------------------------------------------------- 路径
BASE_DIR = Path(__file__).resolve().parent.parent          # RAG_task_book/ 仓库根
RESOURCES_DIR = BASE_DIR / "resources"
PDF_PATH = RESOURCES_DIR / "重点新材料研发及应用国家科技重大专项课题任务书.pdf"

VECTOR_DIR = BASE_DIR / "chromadb_data"                    # ChromaDB 持久化目录
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"  # 提示词模板目录

# ---------------------------------------------------------------- 常量
COLLECTION_NAME = "project_task_book"
DOC_NAME = "《重点新材料研发及应用国家科技重大专项课题任务书》"

BATCH_SIZE = 20        # ⚠️ embedding 模型(qwen3.7-text-embedding-flash)单次请求上限 20 条
CHUNK_SIZE = 500       # 中文按字符数计,500 字符 ≈ 一小段/半张表
CHUNK_OVERLAP = 80     # 片段间重叠,避免表格行被切断后丢失上下文
TOP_K = 3              # 检索工具召回条数
MAX_REWRITES = 1       # 查询改写最多轮数,防止 judge↔rewrite 死循环
SEP = "\n\n---\n\n"    # 工具返回的多片段之间的分隔符(judge 节点靠它拆开)
