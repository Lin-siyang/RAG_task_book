"""提示词加载器 —— 从 rag/prompts/ 目录读取模板并填充占位符。

对应主项目的 utils/prompt_loader.py:提示词外置为文件,业务代码只引用模板名。
"""

from __future__ import annotations

from rag.config import PROMPTS_DIR


def load_prompt(name: str, **ctx: str) -> str:
    """读取 rag/prompts/<name>.md,并用 ctx 填充 {placeholder} 占位符。

    :param name: 模板名(不带扩展名),如 "agent_system"
    :param ctx:  占位符值,如 doc_name="《...》"
    """
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"提示词模板不存在: {path}")
    text = path.read_text(encoding="utf-8")
    return text.format(**ctx) if ctx else text
