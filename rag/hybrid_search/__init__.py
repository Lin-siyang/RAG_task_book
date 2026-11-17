"""hybrid_search 子包 —— 混合检索的两个组件,供 agent/agent_tool.py 的工具函数组合调用。

    bm25_retriever.py  BM25 稀疏检索(jieba 中文分词 + rank-bm25,向量库全量切片建索引)
    reranker.py        BGE 重排(bge-reranker-v2-m3 cross-encoder,对候选按相关性精排)

混合检索流水线(agent_tool.hybrid_search_project_doc):
    向量召回(HYBRID_VECTOR_TOP_K) + BM25 召回(HYBRID_BM25_TOP_K)
    → 合并去重 → BGE 重排取 HYBRID_RERANK_TOP_K → 格式化返回
"""
