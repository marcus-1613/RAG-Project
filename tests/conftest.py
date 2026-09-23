"""pytest 共享夹具。

提供临时目录、mock 的 embedding / LLM 等，确保测试不依赖真实 API Key，
也不会污染真实的 data 目录。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 将项目根目录加入 sys.path，保证 `import src.xxx` 可用
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """返回一个临时目录，模拟 data 目录结构。"""
    (tmp_path / "pdfs").mkdir(exist_ok=True)
    (tmp_path / "chroma").mkdir(exist_ok=True)
    return tmp_path


@pytest.fixture
def sample_documents():
    """返回几个用于切分测试的简单 Document。"""
    from langchain_core.documents import Document

    text = (
        "这是第一段测试文本。"
        "它包含足够多的字符，用于验证文本切分的逻辑是否正确。"
        "我们会把它切分成多个文本块。" * 20
    )
    return [Document(page_content=text, metadata={"source": "test.pdf", "page": 0})]
