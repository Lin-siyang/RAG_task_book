"""RAG_task_book 问答侧核心包 —— RAG 引擎(建库 + 检索问答的基础设施)。

分层(对应主项目 backend/app/rag/ 的工程化约定):
    config/           配置子包(model_config 模型 / prompt_config 提示词 / chroma_config 检索)
    models.factory    Chat / Embedding 模型工厂
    prompts/          提示词模板(经 config.prompt_config.load_prompt 加载)
    document_handler  文件处理分发(检查文件类型 → 返回对应的文件处理函数)
    text_splitter     文档切分(按类型选择切分器类,差异化 chunk 参数/切分方法)
    ingest            建库流水线编排(python -m rag.ingest)
    vector_store      ChromaDB 封装(建库写入 + 持久化加载)
    schemas/          Pydantic 模型类与 LangGraph State

Agent 构建流程与检索工具在兄弟包 agent/ 中(对应主项目 backend/app/agent/)。
"""
