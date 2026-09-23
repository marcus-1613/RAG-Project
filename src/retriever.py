"""检索模块。

从 ChromaDB 获取 retriever，实现 Top-K 相似文本块检索。
"""

from __future__ import annotations

from langchain_core.documents import Document

from src import config
from src.vector_store import get_vector_store


def get_retriever(top_k: int | None = None):
    """获取一个配置好的 retriever。

    Args:
        top_k: 返回的最相似文本块数量，默认取 config.TOP_K。

    Returns:
        Retriever: 可用于检索的 retriever 对象。
    """
    if top_k is None:
        top_k = config.TOP_K

    vector_store = get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": top_k})


def retrieve(query: str, top_k: int | None = None) -> list[Document]:
    """根据用户问题检索 Top-K 个最相似的文本块。

    Args:
        query: 用户问题。
        top_k: 返回数量，默认 config.TOP_K。

    Returns:
        list[Document]: 最相似的文本块列表，每个都保留
            page_content 及 source、page 等 metadata。
    """
    retriever = get_retriever(top_k=top_k)
    return retriever.invoke(query)
