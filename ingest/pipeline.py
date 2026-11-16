"""建库流水线 —— PDF 解析 → 切分 → 向量化 → ChromaDB 持久化。

流程:
    1. PyPDFLoader 逐页解析 PDF,每页一个 Document,metadata 写入 {source, page}(页码从 1 开始)
    2. RecursiveCharacterTextSplitter 递归字符切分(中文标点也作为分隔符,兼容表格文本)
       每个片段自动继承所属页码的 metadata
    3. 按 BATCH_SIZE 分批调用 embedding 并写入 ChromaDB(该模型单次请求最多 20 条)
    4. 持久化存储到 chromadb_data/,问答脚本(rag_agent.py)直接读取,无需重复构建
"""

from __future__ import annotations

import shutil

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import BATCH_SIZE, COLLECTION_NAME, CHUNK_OVERLAP, CHUNK_SIZE, PDF_PATH, VECTOR_DIR
from rag.models import build_embeddings


def load_pages() -> list:
    """逐页解析 PDF,每页一个 Document,metadata: {source: 文件名, page: 页码(从 1 开始)}"""
    loader = PyPDFLoader(str(PDF_PATH))
    pages = loader.load()  # 每页一个 Document,metadata 里自带 page(0 开始)和 source
    # 统一 metadata:source 只留文件名、页码改成人类习惯的从 1 开始
    for idx, doc in enumerate(pages):
        doc.metadata["source"] = PDF_PATH.name
        doc.metadata["page"] = idx + 1
    return pages


def split_pages(pages: list) -> list:
    """递归字符切分;中文表格/段落里常见的分隔符都纳入,尽量按语义边界切"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        add_start_index=True,  # metadata 里附带片段在原文中的起始下标,便于调试
    )
    chunks = splitter.split_documents(pages)
    # split_documents 会继承每页的 {source, page} → 每个片段都带页码索引
    return chunks


def build_vector_store(chunks: list) -> Chroma:
    """分批向量化写入 ChromaDB;重建索引时清空旧目录,保证幂等(重复运行不会叠加重复片段)。"""
    if VECTOR_DIR.exists():
        print(f"⚠️ 已存在 {VECTOR_DIR.name}/,删除旧索引后重建 ...")
        shutil.rmtree(VECTOR_DIR)

    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(VECTOR_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    vs = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=build_embeddings(),
    )

    # ⚠️ 关键:按 BATCH_SIZE 条一批分批写入(该 embedding 模型的单请求上限)
    total = len(chunks)
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        vs.add_documents(batch)
        print(f"  已入库 {min(i + BATCH_SIZE, total)}/{total} 条片段")
    return vs


def run() -> Chroma:
    """完整建库流程(供 ingest_pdf.py 入口调用),含进度打印与检索冒烟测试。"""
    print(f"📄 解析 PDF: {PDF_PATH.name}")
    pages = load_pages()
    print(f"   共 {len(pages)} 页")

    chunks = split_pages(pages)
    print(f"✂️ 切分为 {len(chunks)} 个片段(每片段 metadata 含 source + page)")

    print(f"🔎 分批(BATCH_SIZE={BATCH_SIZE})向量化并写入 {VECTOR_DIR.name}/ ...")
    vs = build_vector_store(chunks)
    print(f"✅ 入库完成,向量库共 {vs._collection.count()} 条")

    # 冒烟测试:检索一次,确认页码索引正常
    print("\n🧪 冒烟测试 —— query='研究内容' top3:")
    for d in vs.similarity_search("研究内容", k=3):
        print(f"   第{d.metadata.get('page')}页: {d.page_content[:60]!r}...")
    return vs
