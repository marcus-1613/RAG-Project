"""检索模块：统一入口，按配置分发到不同的检索策略。

四种模式（可用 RETRIEVAL_MODE 切换，也可在评估脚本中逐个对比）：

    vector         仅向量检索。原始行为，作为基线。
    bm25           仅关键词检索。用于验证"专有名词类问题上 BM25 是否更强"。
    hybrid         向量 + BM25，用 RRF 融合。默认模式。
    hybrid_rerank  在 hybrid 基础上再做 Cross-Encoder 精排。

为什么召回数量（CANDIDATE_K）要大于返回数量（TOP_K）：
    融合与重排都只能对**已有候选**重新排序，无法把没召回的东西变出来。
    如果只召回 5 条再融合，融合就没有腾挪空间——两路检索各自的第一名
    只能原样保留。先多召回、再精排，是这类流水线的通用做法。
"""

from __future__ import annotations

from typing import Iterable

from langchain_core.documents import Document

from src import config
from src.bm25 import BM25Index, build_index
from src.fusion import rrf_fuse
from src.rerank import get_reranker
from src.vector_store import get_all_documents, get_vector_store


# ============================================================
# 向量检索
# ============================================================

def vector_search(query: str, top_k: int) -> list[Document]:
    """向量检索：按语义相似度返回文本块。

    Args:
        query: 用户问题。
        top_k: 返回数量。

    Returns:
        list[Document]: 按相似度降序的文本块。
    """
    store = get_vector_store()
    return store.similarity_search(query, k=top_k)


def get_retriever(top_k: int | None = None):
    """获取一个配置好的**向量** retriever。

    保留此函数是为了向后兼容（原 tests 与脚本会调用它）。
    新的调用方请使用 retrieve() / search()，它们会遵循 RETRIEVAL_MODE。

    Args:
        top_k: 返回数量，默认取 config.TOP_K。

    Returns:
        Retriever: 可调用 invoke 的向量检索器。
    """
    if top_k is None:
        top_k = config.TOP_K
    return get_vector_store().as_retriever(search_kwargs={"k": top_k})


# ============================================================
# BM25 检索（含索引缓存）
# ============================================================

# BM25 索引缓存：键为语料规模，语料变化时重建。
# 语料相同则复用，避免每次提问都重新分词建索引。
_bm25_cache: dict[int, BM25Index] = {}


def clear_retrieval_cache() -> None:
    """清空 BM25 索引缓存。

    必须在**知识库内容变化后**调用（新增 PDF、清空 data 目录等）。
    否则索引仍指向旧语料，检索结果会与新库不一致。
    """
    _bm25_cache.clear()


def get_bm25_index(force_reload: bool = False) -> BM25Index:
    """获取（并缓存）BM25 索引。

    Args:
        force_reload: 为 True 时无条件重建索引。

    Returns:
        BM25Index: 索引对象；语料为空时返回空索引。
    """
    documents = get_all_documents()
    key = len(documents)
    if force_reload or key not in _bm25_cache:
        _bm25_cache.clear()
        _bm25_cache[key] = build_index(documents)
    return _bm25_cache[key]


def bm25_search(query: str, top_k: int) -> list[Document]:
    """BM25 关键词检索。

    Args:
        query: 用户问题。
        top_k: 返回数量。

    Returns:
        list[Document]: 按 BM25 分数降序的文本块。
    """
    index = get_bm25_index()
    return [doc for doc, _score in index.search(query, top_k=top_k)]


# ============================================================
# 统一检索入口
# ============================================================

def search(
    query: str,
    top_k: int | None = None,
    mode: str | None = None,
) -> list[tuple[Document, float]]:
    """按指定模式检索，并返回带分数的结果。

    Args:
        query: 用户问题。
        top_k: 返回数量，默认 config.TOP_K。
        mode: 检索模式，默认 config.RETRIEVAL_MODE。

    Returns:
        list[tuple[Document, float]]: (文本块, 相关性分) 降序列表。
            不同模式的分数**含义不同**（向量模式是相似度、BM25 模式是
            BM25 原始分、融合模式是 RRF 分），因此只可用于同模式内比较，
            不要跨模式比较数值大小。

    Raises:
        ValueError: 模式名非法时抛出。
    """
    mode = config.ensure_valid_mode(mode or config.RETRIEVAL_MODE)
    top_k = top_k or config.TOP_K

    if mode == "vector":
        return [(doc, 0.0) for doc in vector_search(query, top_k)]

    if mode == "bm25":
        index = get_bm25_index()
        return index.search(query, top_k=top_k)

    # hybrid / hybrid_rerank：先多路召回，再融合
    candidates = max(config.CANDIDATE_K, top_k)
    vector_docs = vector_search(query, candidates)
    bm25_docs = [doc for doc, _score in get_bm25_index().search(query, candidates)]
    fused = rrf_fuse([vector_docs, bm25_docs], k=config.RRF_K, top_n=candidates)

    if mode == "hybrid_rerank" and config.RERANK_ENABLED:
        reranked = get_reranker(config.RERANK_MODEL).rerank(
            query, [doc for doc, _score in fused], top_k
        )
        return reranked

    return fused[:top_k]


def retrieve(query: str, top_k: int | None = None) -> list[Document]:
    """按配置的检索模式返回文本块（不含分数）。

    Args:
        query: 用户问题。
        top_k: 返回数量，默认 config.TOP_K。

    Returns:
        list[Document]: 排序后的文本块列表。
    """
    return [doc for doc, _score in search(query, top_k=top_k)]


def describe_mode(mode: str | None = None) -> str:
    """返回模式的界面展示文案。

    Args:
        mode: 检索模式，默认取配置值。

    Returns:
        str: 人类可读的模式说明。
    """
    mode = config.ensure_valid_mode(mode or config.RETRIEVAL_MODE)
    labels = {
        "vector": "向量检索（语义）",
        "bm25": "BM25 关键词检索",
        "hybrid": "混合检索（向量 + BM25，RRF 融合）",
        "hybrid_rerank": "混合检索 + Cross-Encoder 精排",
    }
    return labels[mode]
