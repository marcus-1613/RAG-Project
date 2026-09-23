"""文本切分模块。

使用 RecursiveCharacterTextSplitter 将 Document 切分为较小的文本块（chunk），
切分参数从 config 模块统一读取。
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config


def _create_splitter() -> RecursiveCharacterTextSplitter:
    """创建并返回一个配置好的 RecursiveCharacterTextSplitter。

    Returns:
        RecursiveCharacterTextSplitter: 按 config 中的参数初始化的切分器。
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.CHUNK_SEPARATORS,
        add_start_index=True,  # 记录 chunk 在原文中的起始位置，方便定位来源
    )


def split_documents(documents: list[Document]) -> list[Document]:
    """将 Document 列表切分为文本块。

    Args:
        documents: 待切分的 Document 列表（通常来自 loader 模块）。

    Returns:
        list[Document]: 切分后的文本块列表，每个 chunk 保留原 metadata
            （source、page 等），并额外带上 start_index。
    """
    splitter = _create_splitter()
    return splitter.split_documents(documents)
