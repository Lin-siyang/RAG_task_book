"""建库子包 —— PDF 任务书向量化流水线(与 rag/ 问答子包对称)。

    pipeline.py    流水线实现(解析 → 切分 → 分批向量化 → ChromaDB)
    ingest_pdf.py  命令行入口(python ingest/ingest_pdf.py)

配置(路径/切分参数/模型)复用 rag/config.py 与 rag/models/factory.py,单一来源不重复。
"""
