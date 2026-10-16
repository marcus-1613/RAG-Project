"""BM25 关键词检索的单元测试。

这些测试**不需要** ChromaDB、Embedding 模型或 API Key——
BM25 是纯逻辑，只依赖 jieba 分词与 rank_bm25 打分。
"""

from __future__ import annotations

from langchain_core.documents import Document

from src.bm25 import BM25Index, build_index, tokenize


# ============================================================
# 分词
# ============================================================

def test_tokenize_chinese_is_segmented():
    """中文必须被切成词，而不是按字拆开——这是中文 BM25 的前提。"""
    tokens = tokenize("向量数据库用于存储高维向量")
    assert "向量" in tokens or "数据库" in tokens
    # 分词后词数应明显少于字数
    assert len(tokens) < len("向量数据库用于存储高维向量")


def test_tokenize_keeps_english_words_whole_and_lowercased():
    """英文专有名词必须整体保留并转小写（BM25 的关键价值就在这里）。"""
    tokens = tokenize("ChromaDB 是一个 Vector Database")
    assert "chromadb" in tokens
    assert "vector" in tokens


def test_tokenize_drops_punctuation():
    """标点不应进入索引。"""
    tokens = tokenize("人工智能，是…一门学科。")
    for token in tokens:
        assert token.isalnum(), f"非字母数字词元混入索引：{token!r}"


def test_tokenize_empty_input():
    assert tokenize("") == []
    assert tokenize("   ，。！") == []


# ============================================================
# 索引与检索
# ============================================================

def _corpus() -> list[Document]:
    return [
        Document(
            page_content="ChromaDB 是一个轻量级的开源向量数据库，常用于 RAG 系统。",
            metadata={"source": "a.pdf", "page": 1, "start_index": 0},
        ),
        Document(
            page_content="光合作用是绿色植物利用光能把二氧化碳和水合成有机物的过程。",
            metadata={"source": "b.pdf", "page": 1, "start_index": 0},
        ),
        Document(
            page_content="蜜蜂通过舞蹈向同伴传递蜜源的方向和距离信息。",
            metadata={"source": "c.pdf", "page": 1, "start_index": 0},
        ),
    ]


def test_build_index_length():
    index = build_index(_corpus())
    assert len(index) == 3


def test_search_ranks_keyword_matching_document_first():
    """专有名词查询应精确命中对应片段。"""
    index = BM25Index(_corpus())
    results = index.search("ChromaDB 是什么？", top_k=3)
    assert results, "应至少命中一条"
    assert "ChromaDB" in results[0][0].page_content


def test_search_returns_empty_for_unrelated_query():
    """没有任何共同词元时不应返回结果（否则会把噪声混进融合候选）。"""
    index = BM25Index(_corpus())
    assert index.search("量子纠缠的数学推导", top_k=3) == []


def test_search_filters_out_documents_without_shared_tokens():
    """只返回与问题共享词元的文档，不相关文档必须被排除。"""
    index = BM25Index(_corpus())
    results = index.search("蜜蜂的舞蹈语言", top_k=3)
    contents = [doc.page_content for doc, _score in results]
    assert any("蜜蜂" in text for text in contents)
    assert not any("光合作用" in text for text in contents)


def test_search_respects_top_k():
    index = BM25Index(_corpus())
    # 用一个能在多篇文档命中的问题
    results = index.search("向量 数据库 植物 蜜蜂", top_k=1)
    assert len(results) <= 1


def test_empty_corpus_is_safe():
    """空语料不应抛异常。"""
    index = BM25Index([])
    assert len(index) == 0
    assert index.search("任意问题") == []
