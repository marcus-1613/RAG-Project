"""RRF 融合的单元测试（纯逻辑，无外部依赖）。"""

from __future__ import annotations

from langchain_core.documents import Document

from src.fusion import doc_key, rrf_fuse


def _doc(name: str, start: int = 0) -> Document:
    return Document(
        page_content=f"内容 {name}",
        metadata={"source": "a.pdf", "page": 0, "start_index": start},
    )


# ============================================================
# 文本块标识
# ============================================================

def test_doc_key_prefers_location_metadata():
    key = doc_key(_doc("x", start=42))
    assert key == "a.pdf#0#42"


def test_doc_key_falls_back_to_content_hash():
    doc = Document(page_content="没有 start_index 的片段", metadata={"source": "a.pdf"})
    key = doc_key(doc)
    assert key.startswith("sha1:")
    # 内容相同则标识相同，可用于去重
    assert key == doc_key(
        Document(page_content="没有 start_index 的片段", metadata={"source": "a.pdf"})
    )


# ============================================================
# 融合
# ============================================================

def test_rrf_promotes_document_appearing_in_both_rankings():
    """两路都出现的文本块应排到最前——这是 RRF 的核心作用。"""
    shared = _doc("shared", start=1)
    only_a = _doc("only_a", start=2)
    only_b = _doc("only_b", start=3)

    fused = rrf_fuse([[shared, only_a], [shared, only_b]], top_n=3)
    assert fused[0][0].page_content == "内容 shared"


def test_rrf_keeps_documents_unique():
    """同一文本块在多路中出现多次，只应保留一条。"""
    shared = _doc("shared", start=1)
    fused = rrf_fuse([[shared], [shared], [shared]], top_n=10)
    assert len(fused) == 1


def test_rrf_top_n_limits_output():
    ranking = [_doc(f"d{i}", start=i) for i in range(10)]
    fused = rrf_fuse([ranking], top_n=3)
    assert len(fused) == 3


def test_rrf_scores_are_ordered_descending():
    ranking_a = [_doc(f"a{i}", start=i) for i in range(5)]
    ranking_b = [_doc(f"b{i}", start=100 + i) for i in range(5)]
    fused = rrf_fuse([ranking_a, ranking_b], top_n=10)
    scores = [score for _doc_, score in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_handles_empty_rankings():
    assert rrf_fuse([[], []]) == []
    assert rrf_fuse([]) == []
