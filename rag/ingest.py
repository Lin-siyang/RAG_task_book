"""建库流水线编排 —— 文件处理(document_handler) → 切分(text_splitter) → 向量化写入(vector_store)。

三个脚本的分工:
    document_handler.py  检查文件类型,返回对应的文件处理函数(怎么读进来)
    text_splitter.py     按文件类型选择切分器类(怎么切开)
    vector_store.py      ChromaDB 封装(本脚本调用它完成向量化与写入)

跑法:
    python -m rag.ingest          # 在仓库根目录运行;仅首次建库,或文档更新后重建
"""

from __future__ import annotations

from langchain_chroma import Chroma

from rag.config import PDF_PATH
from rag.config.model_config import BATCH_SIZE
from rag.document_handler import get_file_handler
from rag.text_splitter import get_splitter
from rag.vector_store import build_vector_store


def run(pdf_path=PDF_PATH) -> Chroma:
    """完整建库流程(默认任务书 PDF,可传其他文档路径),含进度打印与检索冒烟测试。"""
    # 1. 文件处理:检查类型 → 拿到对应处理函数 → 解析
    file_type, handler = get_file_handler(pdf_path)
    print(f"📄 解析文档: {pdf_path.name} (类型 {file_type})")
    docs = handler(pdf_path)

    # 2. 切分:按文件类型选择切分器
    chunks = get_splitter(file_type).split(docs)
    print(f"✂️ 切分为 {len(chunks)} 个切片(每个切片 metadata 含 source + page)")

    # 3. 向量化与写入:调用 vector_store.py,按 BATCH_SIZE 分批(该 embedding 模型单次请求上限)
    print(f"🔎 分批(BATCH_SIZE={BATCH_SIZE})向量化并写入 chromadb_data/ ...")
    vs = build_vector_store(chunks)
    print(f"✅ 入库完成,向量库共 {vs._collection.count()} 条")

    # 冒烟测试:检索一次,确认页码索引正常
    print("\n🧪 冒烟测试 —— query='研究内容' top3:")
    for d in vs.similarity_search("研究内容", k=3):
        print(f"   第{d.metadata.get('page')}页: {d.page_content[:60]!r}...")
    return vs


if __name__ == "__main__":
    run()
