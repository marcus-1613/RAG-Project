"""向量存储模块。

负责创建、持久化 ChromaDB，以及向其中添加文档、查询已有文档等操作。
ChromaDB 数据持久化到 config.CHROMA_DIR，重启后无需重新 Embedding。
"""

from __future__ import annotations

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src import config
from src.embeddings import get_embeddings


def get_vector_store() -> Chroma:
    """获取（或创建）持久化的 ChromaDB 向量存储。

    若 config.CHROMA_DIR 下已有数据则直接复用，否则创建新的空库。

    Returns:
        Chroma: 持久化的向量存储对象。
    """
    # 确保持久化目录存在
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name="pdf_knowledge_base",
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )


def add_documents(documents: list[Document]) -> None:
    """将文档（文本块）写入向量存储。

    Args:
        documents: 已切分好的文本块列表。
    """
    vector_store = get_vector_store()
    vector_store.add_documents(documents)


def get_existing_sources() -> set[str]:
    """返回向量库中已入库的文档来源（文件名）集合。

    用于判断某个 PDF 是否已经处理过，避免重复 Embedding。

    Returns:
        set[str]: 已入库的 source（文件名）集合。
    """
    try:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        collection = client.get_collection("pdf_knowledge_base")
        all_metadata = collection.get(include=["metadatas"])
        sources: set[str] = set()
        for meta in all_metadata.get("metadatas", []):
            if meta and meta.get("source"):
                sources.add(meta["source"])
        return sources
    except Exception:  # noqa: BLE001 - 库不存在或为空时返回空集合
        return set()
