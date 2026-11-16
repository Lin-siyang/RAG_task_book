"""问答入口 —— LangGraph 构建的项目书 RAG 问答顾问(带检索工具 + 检索质量校验 + 查询改写)。

跑法:
    python rag_agent.py                # 默认演示问题
    python rag_agent.py "你的问题"     # 自定义问题

本文件只保留「编译图 + 流式打印」两部分;图结构、节点实现、提示词见 rag/ 包。
模块级 GRAPH / SEP 是对外稳定接口 —— evals/run_ragas_eval.py 用 importlib 按路径
加载本文件,依赖这两个属性与导入副作用(载 .env / 连向量库 / 编译图)。
"""

from __future__ import annotations

import sys
from pathlib import Path

# evals/run_ragas_eval.py 用 importlib 按文件路径加载本模块,此时 sys.path 里没有仓库根,
# 需手动把本目录加入 sys.path,否则 import rag 会失败(python rag_agent.py 直跑时此行为无副作用)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain.messages import HumanMessage  # noqa: E402

from rag.config import SEP  # noqa: E402  eval 依赖的模块级片段分隔符
from rag.graph import build_graph  # noqa: E402

GRAPH = build_graph()

# ---------------------------------------------------------------- 流式输出
NODE_LABELS = {"agent": "\n🤖 顾问", "rewrite": "\n🔁 查询改写", "generate": "\n📝 回答"}


def stream_answer(question: str) -> None:
    """逐 token 流式打印模型输出(judge 节点的结构化输出不打印)。"""
    print(f"\n{'=' * 70}\n❓ {question}\n{'-' * 70}")
    current_node = None
    for chunk, meta in GRAPH.stream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="messages",   # 按 token 流式返回
    ):
        node = meta.get("langgraph_node")
        content = chunk.content
        if node not in NODE_LABELS or not isinstance(content, str) or not content:
            continue  # 跳过 judge 的 JSON token、工具调用参数 token 等
        if node != current_node:
            print(NODE_LABELS[node], end=":", flush=True)
            current_node = node
        print(content, end="", flush=True)
    print()


if __name__ == "__main__":
    questions = sys.argv[1:] or [
        "参与这个课题的单位有哪些?",      # 需要检索任务书
        "你好,介绍一下你自己",            # 无需检索,直接回答
    ]
    for q in questions:
        stream_answer(q)
