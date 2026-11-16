"""RAG_task_book 问答侧核心包 —— LangGraph RAG 顾问的共享实现。

分层(对应主项目 backend/app 的工程化约定):
    config          配置单一来源(.env 加载 / 路径 / 常量)
    models.factory  Chat / Embedding 模型工厂
    prompts/        提示词模板(经 prompt_loader 加载)
    vector_store    ChromaDB 持久化加载
    tools           @tool 检索工具
    schemas         Pydantic 校验模型 + LangGraph State
    graph           LangGraph 五节点图构建

建库流水线在兄弟包 ingest/ 中(与问答侧解耦)。
"""
