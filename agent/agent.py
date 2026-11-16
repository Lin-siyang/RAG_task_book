"""agent 创建流程 —— agent / tools / judge / rewrite / generate 五节点 RAG 问答顾问。

图结构:
                    ┌──────────┐  无 tool_calls(闲聊/常识,直接回答)
     START ────────▶│  agent   │────────────────────────────────▶ END
                    └────┬─────┘  有 tool_calls
                         ▼
                    ┌──────────┐
                    │  tools   │  执行 search_project_doc(top_k=TOP_K)
                    └────┬─────┘
                         ▼
                    ┌──────────┐  Pydantic 结构化输出逐片段评 grade(yes/no)
                    │  judge   │  ≥1 个 yes ──────────────┐
                    └────┬─────┘                          ▼
                         │ 全部 no 且未改写过        ┌───────────┐
                         ▼                          │ generate  │──▶ END
                    ┌──────────┐   改写后回到 agent  │(最终回答) │
                    │ rewrite  │─────────▶ agent    └───────────┘
                    └──────────┘   (改写过仍失败 → 直接进 generate 兜底)

- 所有对话消息都存在 MessagesState 的 messages 里;另加 relevant_chunks / rewrite_count
  两个辅助字段在节点间传递"被判为有效的片段"和"改写次数"。
- judge 节点用 llm.with_structured_output(RelevanceGrade) 强制模型只能输出 yes / no。
"""

from __future__ import annotations

from typing import Literal

from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.agent_tool import search_project_doc
from rag.config.chroma_config import MAX_REWRITES, SEP
from rag.config.prompt_config import DOC_NAME, load_prompt
from rag.models import make_llm
from rag.schemas.models import RagState, RelevanceGrade

# 提示词统一从 rag/prompts/ 加载(外置模板,便于调优与评审)
AGENT_SYSTEM = load_prompt("agent_system", doc_name=DOC_NAME)
JUDGE_SYSTEM = load_prompt("judge_system")
REWRITE_SYSTEM = load_prompt("rewrite_system")
GENERATE_SYSTEM = load_prompt("generate_system", doc_name=DOC_NAME)


# ---------------------------------------------------------------- 节点 1:agent(带检索工具)
def agent_node(state: RagState) -> dict:
    llm = make_llm().bind_tools([search_project_doc])
    response = llm.invoke([SystemMessage(AGENT_SYSTEM)] + state["messages"])
    return {"messages": [response]}


def route_after_agent(state: RagState) -> Literal["tools", "__end__"]:
    """模型决定:调了工具 → 走 tools 节点;直接回答 → 结束。"""
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


# ---------------------------------------------------------------- 节点 2:judge(结构化输出校验)
def judge_node(state: RagState) -> dict:
    """用 Pydantic 结构化输出逐个评估召回的 TOP_K 个片段,只保留 grade=yes 的。"""
    # 工具结果是最后一条 ToolMessage
    tool_msg = next((m for m in reversed(state["messages"]) if isinstance(m, ToolMessage)), None)
    chunks = (tool_msg.content or "").split(SEP) if tool_msg else []

    init_question = state["messages"][0].content  # 始终以最初的问题为评判基准
    # method="function_calling":DeepSeek 不支持 json_schema 类型的 response_format,
    # 走 function calling 的方式让模型按规定 schema 输出 yes/no
    llm = make_llm().with_structured_output(RelevanceGrade, method="function_calling")

    relevant: list[str] = []
    for idx, chunk in enumerate(chunks, 1):
        chunk = chunk.strip()
        if not chunk:
            continue
        # 去掉 【片段N | 第X页】 头部行,只评估正文
        body = "\n".join(chunk.splitlines()[1:]) if chunk.startswith("【片段") else chunk
        grade = llm.invoke(
            [SystemMessage(JUDGE_SYSTEM),
             HumanMessage(f"用户问题:{init_question}\n\n检索片段:\n{body}")]
        )
        print(f"  [judge] 片段 {idx}/{len(chunks)} → {grade.grade}")
        if grade.grade == "yes":
            relevant.append(chunk)

    return {"relevant_chunks": relevant}


def route_after_judge(state: RagState) -> Literal["generate", "rewrite"]:
    """有任一片段 yes → 生成回答;全部 no → 改写查询(最多 MAX_REWRITES 轮,超限兜底生成)。"""
    if state.get("relevant_chunks"):
        return "generate"
    if state.get("rewrite_count", 0) >= MAX_REWRITES:
        print("  [flow] 已达改写上限,用现有结果兜底生成")
        return "generate"
    return "rewrite"


# ---------------------------------------------------------------- 节点 3:rewrite_query
def rewrite_node(state: RagState) -> dict:
    """拿出 state["messages"][0].content 作为 initial question,让模型 improve。"""
    init_question = state["messages"][0].content
    llm = make_llm()
    rewritten = llm.invoke(
        [SystemMessage(REWRITE_SYSTEM), HumanMessage(f"原始问题:{init_question}")]
    ).content.strip()
    print(f"  [rewrite] {init_question!r} → {rewritten!r}")
    # 改写后作为一条新的用户消息进入 agent,促使它用新查询再次调用检索工具
    return {
        "messages": [HumanMessage(content=f"(改写后的查询)请重新检索并回答:{rewritten}")],
        "rewrite_count": state.get("rewrite_count", 0) + 1,
        "relevant_chunks": [],  # 清空上轮结果
    }


# ---------------------------------------------------------------- 节点 4:generate_answer
def generate_node(state: RagState) -> dict:
    """拿出 initial question + 最终有效片段,让模型严格依据资料回答。"""
    init_question = state["messages"][0].content
    chunks = state.get("relevant_chunks") or []
    context = SEP.join(chunks) if chunks else "(本次未检索到有效片段)"

    llm = make_llm()
    response = llm.invoke([
        SystemMessage(GENERATE_SYSTEM),
        HumanMessage(f"原始问题:\n{init_question}\n\n资料片段:\n{context}"),
    ])
    return {"messages": [response]}


# ---------------------------------------------------------------- 组图
def build_graph():
    g = StateGraph(RagState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode([search_project_doc]))
    g.add_node("judge", judge_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("generate", generate_node)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent)          # tools / END
    g.add_edge("tools", "judge")
    g.add_conditional_edges("judge", route_after_judge)          # generate / rewrite
    g.add_edge("rewrite", "agent")
    g.add_edge("generate", END)
    return g.compile()
