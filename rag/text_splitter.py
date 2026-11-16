"""文档切分 —— 以类为组织单位,针对不同文件类型使用不同的 chunk 参数/切分方法。

与 document_handler.py 的分工:
    document_handler.py  负责"怎么读进来"(检查类型 → 返回文件处理函数)
    text_splitter.py     负责"读进来后怎么切开"(按类型选择切分器类)

扩展方式:新增类型时继承 BaseSplitter 写一个子类(差异化 chunk_size / 分隔符,
甚至覆盖 split() 换切分方法,如未来引入 MarkdownHeaderTextSplitter),
再到 SPLITTERS 注册一行。当前各类型的参数先写成一样的(取自 chroma_config),
后续按检索效果逐类型调优。
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config.chroma_config import CHUNK_OVERLAP, CHUNK_SIZE


# ---------------------------------------------------------------- 切分器基类
class BaseSplitter:
    """切分器基类:统一封装"参数 → RecursiveCharacterTextSplitter → 切分"的流程。

    子类通过覆盖类属性(chunk_size / chunk_overlap / separators)实现按类型差异化;
    需要完全不同的切分方法时,覆盖 split() 即可。
    """

    file_type: str = ""                  # 本切分器对应的文件类型(注册用)
    chunk_size: int = CHUNK_SIZE         # 中文按字符数计,500 字符 ≈ 一小段/半张表
    chunk_overlap: int = CHUNK_OVERLAP   # 片段间重叠,避免表格行被切断后丢失上下文
    separators: tuple[str, ...] = ("\n\n", "\n", "。", "；", "，", " ", "")
    add_start_index: bool = True         # metadata 附带切片在原文中的起始下标,便于调试

    def build(self) -> RecursiveCharacterTextSplitter:
        """由当前类属性构建 LangChain 切分器实例。"""
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=list(self.separators),
            add_start_index=self.add_start_index,
        )

    def split(self, docs: list) -> list:
        """切分文档;split_documents 自动让每个切片继承 {source, page}。"""
        return self.build().split_documents(docs)


# ---------------------------------------------------------------- 按类型的切分器(参数当前一致,后续可差异化)
class PdfSplitter(BaseSplitter):
    """PDF(中文任务书):按段落/标点切。"""

    file_type = ".pdf"


class MarkdownSplitter(BaseSplitter):
    """Markdown:在相同 chunk 参数基础上,把标题层级加入分隔符(优先沿标题切)。"""

    file_type = ".md"
    separators = ("\n# ", "\n## ", "\n### ", "\n\n", "\n", "。", "；", "，", " ", "")


class TextSplitter(BaseSplitter):
    """纯文本:与 pdf 相同的中文标点切分。"""

    file_type = ".txt"


# ---------------------------------------------------------------- 注册表:扩展名 → 切分器类
SPLITTERS: dict[str, type[BaseSplitter]] = {
    ".pdf": PdfSplitter,
    ".md": MarkdownSplitter,
    ".txt": TextSplitter,
}


def get_splitter(file_type: str) -> BaseSplitter:
    """按文件类型取切分器实例;未注册类型回退纯文本切分器。"""
    return SPLITTERS.get(file_type, TextSplitter)()
