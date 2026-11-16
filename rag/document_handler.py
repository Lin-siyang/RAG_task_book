"""文件处理分发 —— 检查文件类型(pdf/md/txt...),返回对应的文件处理(解析)函数。

职责边界:本脚本只负责"识别类型 + 给出处理函数",不做切分(见 text_splitter.py)、
不做向量化写入(见 vector_store.py / ingest.py)。

新增文件类型:写一个 load_xxx 处理函数(负责解析 + metadata 规范化) + 在 LOADERS 注册一行。
metadata 约定:所有处理函数产出的 Document 统一带 {source: 文件名, page: 页码(从 1 开始)},
无页码概念的纯文本类型 page 固定为 1 —— 问答侧(agent_tool.py)据此标注来源页码。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader


# ---------------------------------------------------------------- 各类型的文件处理函数
def load_pdf(path: str | Path) -> list:
    """PDF 逐页解析;PyPDFLoader 的 page 从 0 开始,统一改成人类习惯的从 1 开始。"""
    path = Path(path)
    pages = PyPDFLoader(str(path)).load()
    for idx, doc in enumerate(pages):
        doc.metadata["source"] = path.name
        doc.metadata["page"] = idx + 1
    return pages


def load_text(path: str | Path) -> list:
    """纯文本(txt/md)整篇解析;无页码概念,page 固定为 1。"""
    path = Path(path)
    doc = TextLoader(str(path), encoding="utf-8").load()[0]
    doc.metadata["source"] = path.name
    doc.metadata["page"] = 1
    return [doc]


# ---------------------------------------------------------------- 注册表:扩展名 → 文件处理函数
LOADERS: dict[str, Callable[[str | Path], list]] = {
    ".pdf": load_pdf,
    ".txt": load_text,
    ".md": load_text,
}

FALLBACK_TYPE = ".txt"  # 未注册类型的回退处理方式


def detect_file_type(path: str | Path) -> str:
    """检查文件类型:返回标准化扩展名;未注册的类型回退为 .txt 并打警告。"""
    path = Path(path)
    file_type = path.suffix.lower()
    if file_type not in LOADERS:
        print(f"⚠️ 未注册的文件类型 {file_type!r},回退按纯文本处理:{path.name}")
        file_type = FALLBACK_TYPE
    return file_type


def get_file_handler(path: str | Path) -> tuple[str, Callable[[str | Path], list]]:
    """检查文件类型,返回 (类型, 对应的文件处理函数) —— 本脚本对外的核心 API。"""
    file_type = detect_file_type(path)
    return file_type, LOADERS[file_type]


def load_document(path: str | Path) -> tuple[str, list]:
    """便捷入口:检查类型 → 调用对应处理函数完成解析,返回 (类型, Document 列表)。"""
    file_type, handler = get_file_handler(path)
    return file_type, handler(path)
