"""建库入口 —— 把《重点新材料研发及应用国家科技重大专项课题任务书》PDF 建成向量库。

跑法:
    python ingest/ingest_pdf.py   # 仅首次运行,或 PDF 更新后重建

实现细节见 ingest/pipeline.py(解析 → 切分 → 分批向量化 → ChromaDB 持久化),
参数配置在 rag/config.py(chunk_size / overlap / BATCH_SIZE 等)。
"""

from __future__ import annotations

import sys
from pathlib import Path

# rag/ 与 ingest/ 包都在仓库根,直跑/被按路径加载时把仓库根加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.pipeline import run  # noqa: E402

if __name__ == "__main__":
    run()
