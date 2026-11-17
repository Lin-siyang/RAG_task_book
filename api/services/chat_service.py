"""聊天服务层 —— 调用 agent/ 包的 LangGraph RAG 顾问,向上提供两种问答形态。

    warmup()      服务启动时调用:连接知识库向量库、编译 Agent 图(此后所有请求复用)
    chat()        一次性问答:invoke 图,返回完整回答
    chat_stream() 流式问答:stream 图(stream_mode="messages"),把 Agent 输出转成 SSE 事件
"""

from __future__ import annotations

import json
from collections.abc import Generator

from langchain.messages import HumanMessage

# 只有"回答价值"的节点:agent(闲聊直答)与 generate(最终回答);
# judge 输出结构化 JSON、rewrite 输出改写后的查询、tools 输出工具结果,都不应进入回答流
ANSWER_NODES = {"agent", "generate"}
RECURSION_LIMIT = 50  # judge↔rewrite 循环上限,与 evals 适配器保持一致

_GRAPH = None  # 编译好的 LangGraph 图(进程内单例,warmup() 后可用)


def warmup() -> None:
    """服务启动时加载知识库、准备 Agent。

    放在函数内 import:本模块被导入时保持轻量,
    重资源(向量库连接、提示词加载、图编译)集中到 main.py 的启动阶段。
    """
    global _GRAPH
    from agent.agent import build_graph  # 导入 agent 包即连向量库(agent_tool 的加载副作用)
    from rag.hybrid_search.reranker import get_reranker

    _GRAPH = build_graph()
    get_reranker()


def _graph():
    if _GRAPH is None:
        raise RuntimeError("Agent 未初始化:请通过 main.py 的 lifespan 调用 warmup() 后再请求")
    return _GRAPH


def chat(question: str) -> str:
    """一次性问答:返回完整回答文本(最后一条 AIMessage,即 generate / agent 直答的输出)。"""
    result = _graph().invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": RECURSION_LIMIT},
    )
    return result["messages"][-1].content


def chat_stream(question: str) -> Generator[str, None, None]:
    """流式问答:逐 token 产出 SSE 数据行,供路由层包成 StreamingResponse。

    事件协议(SSE,每条以 data: 开头、空行结尾):
        {"type": "response", "content": "..."}   回答片段(逐 token)
        {"type": "error",    "content": "..."}   服务端处理出错
        {"type": "done"}                          结束标记
    """
    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    try:
        for chunk, meta in _graph().stream(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": RECURSION_LIMIT},
            stream_mode="messages",  # 按 token 流式返回
        ):
            node = meta.get("langgraph_node")
            content = chunk.content
            if node not in ANSWER_NODES or not isinstance(content, str) or not content:
                continue  # 跳过 judge 的 JSON token、工具调用参数、rewrite 的改写查询 token 等
            yield sse({"type": "response", "content": content})
        yield sse({"type": "done"})
    except Exception as e:  # 单次请求失败不拖垮连接:向前端报错后正常收尾
        yield sse({"type": "error", "content": f"服务处理出错: {e}"})
        yield sse({"type": "done"})
