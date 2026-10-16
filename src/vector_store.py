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

# 集合名集中定义，避免多处硬编码字符串不一致
COLLECTION_NAME = "pdf_knowledge_base"


def get_vector_store() -> Chroma:
    """获取（或创建）持久化的 ChromaDB 向量存储。

    若 config.CHROMA_DIR 下已有数据则直接复用，否则创建新的空库。

    Returns:
        Chroma: 持久化的向量存储对象。
    """
    # 确保持久化目录存在
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=COLLECTION_NAME,
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


def get_all_documents() -> list[Document]:
    """读取向量库中的全部文本块。

    用途：为 BM25 建立索引。BM25 是**全局**统计量（IDF 依赖整个语料），
    因此不能像向量检索那样只查局部，必须拿到全量文本块。

    语料规模在个人知识库场景下通常是几百到几千个 chunk，
    一次性读入内存是可接受的。若未来语料量级增长到十万以上，
    应改为持久化的倒排索引（如 Elasticsearch / OpenSearch）。

    Returns:
        list[Document]: 全部文本块；库不存在或为空时返回空列表。
    """
    try:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        collection = client.get_collection(COLLECTION_NAME)
        data = collection.get(include=["documents", "metadatas"])
    except Exception:  # noqa: BLE001 - 库不存在或为空时视为没有数据
        return []

    contents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    documents: list[Document] = []
    for index, content in enumerate(contents):
        if not content:
            continue
        metadata = metadatas[index] if index < len(metadatas) else {}
        documents.append(
            Document(page_content=content, metadata=metadata or {})
        )
    return documents


def get_existing_sources() -> set[str]:
    """返回向量库中已入库的文档来源（文件名）集合。

    用于判断某个 PDF 是否已经处理过，避免重复 Embedding。

    Returns:
        set[str]: 已入库的 source（文件名）集合。
    """
    try:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        collection = client.get_collection(COLLECTION_NAME)
        all_metadata = collection.get(include=["metadatas"])
        sources: set[str] = set()
        for meta in all_metadata.get("metadatas", []):
            if meta and meta.get("source"):
                sources.add(meta["source"])
        return sources
    except Exception:  # noqa: BLE001 - 库不存在或为空时返回空集合
        return set()
