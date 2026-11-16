"""提示词配置 —— 模板目录与加载器(读取 prompts/*.txt,支持 {placeholder} 填充)。

对应主项目的 utils/prompt_loader.py:提示词外置为文件,业务代码只引用模板名。
"""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"   # rag/prompts/

# 文档名称:作为 {doc_name} 占位符注入 agent / generate 系统提示词
DOC_NAME = "《重点新材料研发及应用国家科技重大专项课题任务书》"


def load_prompt(name: str, **ctx: str) -> str:
    """读取 rag/prompts/<name>.txt,并用 ctx 填充 {placeholder} 占位符。

    :param name: 模板名(不带扩展名),如 "agent_system"
    :param ctx:  占位符值,如 doc_name=DOC_NAME
    """
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"提示词模板不存在: {path}")
    text = path.read_text(encoding="utf-8")
    return text.format(**ctx) if ctx else text
