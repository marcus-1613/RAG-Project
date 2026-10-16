"""检索结果融合模块（RRF）。

要解决的核心问题：**两路检索的分数不可比。**

    向量检索返回的是余弦相似度，取值范围 0~1；
    BM25 返回的是原始分，没有上界，而且数值高度依赖语料的统计特征
    （文档数、平均长度、词频分布）。

    如果直接把两者加权求和（比如 `0.7 * 向量分 + 0.3 * BM25 分`），
    结果会完全被 BM25 的量级主导，权重就变成了假的。要做加权，就得先
    归一化——而归一化窗口怎么选又是个新的主观决定，且在不同语料上不通用。

RRF（Reciprocal Rank Fusion）绕开了这个问题：**它只看排名，不看分数。**

    对每个检索器返回的列表，第 rank 名的贡献是 1 / (k + rank)，
    然后把同一个文本块在各路中的贡献累加。

    这样每路检索器各贡献"一份排名"，量纲差异自然消失，也不需要调权重。
    k 取 60 是原论文的经验值：它的作用是压低"绝对第一名"的优势，
    让排名靠前但没到第一的结果也有机会被融合进来。

参考：Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and
individual Rank Learning Methods" (SIGIR 2009)。
"""

from __future__ import annotations

import hashlib

from langchain_core.documents import Document

# RRF 公式中的平滑常数，取原论文经验值 60
DEFAULT_RRF_K = 60


def doc_key(doc: Document) -> str:
    """为文本块生成稳定的唯一标识，用于跨检索结果去重。

    优先使用 (来源, 页码, 起始位置) —— 这三个字段由 loader 和 splitter
    写入 metadata，能唯一定位一个 chunk。缺失时退化为内容哈希。

    Args:
        doc: 文本块。

    Returns:
        str: 稳定的标识字符串。
    """
    meta = doc.metadata or {}
    if "start_index" in meta:
        return f"{meta.get('source', '')}#{meta.get('page', '')}#{meta['start_index']}"
    digest = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()
    return f"sha1:{digest}"


def rrf_fuse(
    rankings: list[list[Document]],
    k: int = DEFAULT_RRF_K,
    top_n: int | None = None,
) -> list[tuple[Document, float]]:
    """将多路检索结果按 RRF 融合成一个排序列表。

    Args:
        rankings: 多路检索结果，每路是一个已按相关性降序排列的文本块列表。
        k: RRF 平滑常数，默认 60。
        top_n: 截断长度，None 表示全部返回。

    Returns:
        list[tuple[Document, float]]: (文本块, RRF 融合分) 列表，
            按融合分降序；同分时按标识排序，保证结果稳定可复现。
    """
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            key = doc_key(doc)
            documents.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if top_n is not None:
        ordered = ordered[:top_n]

    return [(documents[key], score) for key, score in ordered]
