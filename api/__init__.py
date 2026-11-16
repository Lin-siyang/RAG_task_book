"""API 服务包 —— 把 agent/ 包的 RAG 问答能力通过 HTTP 暴露给前端。

    main.py           FastAPI 应用装配(CORS / lifespan 启动加载 / 注册路由)
    routers/chat.py   网址与请求方式(health + 一次性问答 + 流式问答)
    schemas/chat.py   接口数据格式(FastAPI 自动校验)
    services/chat_service.py  服务层(调用 LangGraph Agent,SSE 事件转换)
"""
