"""BM25 关键词检索模块。

为什么需要它：
    向量检索擅长"语义相近"，但它在**必须精确命中**的查询上反而容易漏——
    比如专有名词、型号、代码标识符、缩写（"ChromaDB"、"Transformer"）。
    这类词经过 Embedding 之后语义信息被稀释，容易和一堆泛泛的段落混在一起。

    BM25 基于词频（TF）与逆文档频率（IDF）打分：一个词在越少的文档里出现、
    在当前文档里出现越多次，权重就越高。它不理解语义，但对"精确匹配"
    极其敏感——正好和向量检索互补。

为什么必须分词：
    BM25 是**词级别**的算法，而中文没有天然空格分隔。因此先用 jieba 分词，
    再建索引。这也是中文 BM25 与英文最本质的差别。
"""

from __future__ import annotations

from typing import Any

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """把文本切成 BM25 需要的词元列表。

    中文用 jieba 分词，英文/数字统一转小写后整体保留（避免把
    "ChromaDB" 这类专有名词切断，它们正是 BM25 最该发挥作用的场景）。
    标点与空白直接丢弃。

    **单个汉字会被丢弃。** 中文里的单字绝大多数是虚词（的、是、了、在、
    和……），它们几乎出现在每一段文本里，既没有区分度，又会让"是否共享
    词元"这个判据失效——任意两段中文都会因为共享一个"的"而被判为相关，
    关键词检索就退化成了不过滤。这是中文 BM25 与英文最不一样的地方之一，
    英文里单字母同样没有信息量，但由于英文天然以空格分词，这个问题不突出。

    Args:
        text: 待切分的文本。

    Returns:
        list[str]: 词元列表；输入为空时返回空列表。
    """
    if not text:
        return []

    tokens: list[str] = []
    for piece in jieba.lcut(text):
        piece = piece.strip()
        if not piece:
            continue
        # 非字母数字（标点、空白）直接丢弃
        if not piece.isalnum():
            continue
        # 丢弃单个汉字（虚词为主，无区分度）
        if len(piece) == 1 and not piece.isascii():
            continue
        tokens.append(piece.lower())
    return tokens


class BM25Index:
    """构建在内存中的 BM25 索引。

    索引是对现有语料的一次性计算结果：语料不变时复用即可。
    语料变化（新增 PDF）后需要重建，见 retriever 中的缓存失效逻辑。
    """

    def __init__(self, documents: list[Document]) -> None:
        """用一批文本块构建索引。

        Args:
            documents: 语料中的所有文本块。
        """
        self.documents: list[Document] = list(documents)
        self._tokenized_corpus: list[list[str]] = [
            tokenize(doc.page_content) for doc in self.documents
        ]
        # 语料为空时不构建 BM25 对象（BM25Okapi 会因除零而异常）
        self._bm25: BM25Okapi | None = (
            BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None
        )

    def __len__(self) -> int:
        return len(self.documents)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
        """按 BM25 分数检索最相关的文本块。

        Args:
            query: 用户问题。
            top_k: 返回数量。

        Returns:
            list[tuple[Document, float]]: (文本块, BM25 分数) 列表，
                按分数从高到低排序。只会返回**与问题至少共享一个词元**的
                文本块；完全不共享词元的片段对关键词检索没有意义。
        """
        if self._bm25 is None or top_k <= 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: Any = self._bm25.get_scores(query_tokens)

        # 用"是否共享词元"作为准入门槛，而不是"分数 > 0"。
        #
        # 原因：Okapi BM25 的 IDF 在小语料上可能为**负值**——
        # 公式是 log((N - df + 0.5) / (df + 0.5))，当某个词在多数文档里
        # 都出现时（df 接近 N），对数内小于 1，IDF 变负。rank_bm25 会用
        # "平均 IDF 的 0.25 倍"替换负值，但如果整个语料的平均 IDF 本身
        # 就是负的，替换后依然为负，于是所有分数都为负——这时用
        # "分数 > 0"过滤会把全部候选滤光，表现为"检索不到任何结果"。
        #
        # 共享词元是一个精确且不受 IDF 符号影响的判据。
        query_token_set = set(query_tokens)
        candidates = [
            (doc, float(score))
            for doc, tokens, score in zip(
                self.documents, self._tokenized_corpus, scores
            )
            if query_token_set.intersection(tokens)
        ]
        candidates.sort(key=lambda pair: pair[1], reverse=True)
        return candidates[:top_k]


def build_index(documents: list[Document]) -> BM25Index:
    """构建 BM25 索引（BM25Index 的便捷工厂函数）。

    Args:
        documents: 语料中的所有文本块。

    Returns:
        BM25Index: 已构建好的索引对象。
    """
    return BM25Index(documents)
