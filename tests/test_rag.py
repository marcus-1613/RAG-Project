"""RAG 核心流程测试。

覆盖：配置读取、文本切分、PDF 去重、以及检索分发逻辑。
需要真实 API / 模型的步骤（embedding、LLM 调用）使用 mock，
保证没有 API Key 也能运行全部测试。

检索层的专项测试（BM25 / RRF / 评估 / 重排）见 tests/test_bm25.py 等文件。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src import config
from src.dedup import compute_file_hash
from src.splitter import split_documents


# ============================================================
# 配置读取测试
# ============================================================

def test_config_defaults():
    """验证配置默认值是否正确读取。"""
    assert config.CHUNK_SIZE == 1000
    assert config.CHUNK_OVERLAP == 200
    assert config.TOP_K == 5
    assert config.CANDIDATE_K >= config.TOP_K, "候选数必须不小于返回数，否则融合没有腾挪空间"
    assert config.RETRIEVAL_MODE in config.VALID_RETRIEVAL_MODES
    assert config.DEEPSEEK_MODEL == "deepseek-chat"
    assert config.EMBEDDING_MODEL == "BAAI/bge-small-zh-v1.5"


def test_config_paths_exist():
    """验证 PDF 目录和 Chroma 目录配置存在。"""
    assert isinstance(config.PDF_DIR, Path)
    assert isinstance(config.CHROMA_DIR, Path)


def test_ensure_api_key_missing():
    """验证 API Key 缺失时抛出清晰的中文错误。"""
    if config.DEEPSEEK_API_KEY:
        pytest.skip("当前环境已配置 API Key，跳过缺失场景测试")
    with pytest.raises(ValueError, match="API Key"):
        config.ensure_api_key()


def test_ensure_valid_mode_rejects_unknown_value():
    """非法模式必须立即报错——静默回退会让人以为在跑 hybrid，实际跑的是 vector。"""
    assert config.ensure_valid_mode("HYBRID ") == "hybrid"
    with pytest.raises(ValueError, match="无效的检索模式"):
        config.ensure_valid_mode("hybird")


# ============================================================
# 文本切分测试
# ============================================================

def test_split_documents(sample_documents):
    """验证切分能生成多个 chunk，且保留 metadata。"""
    chunks = split_documents(sample_documents)
    assert len(chunks) >= 1

    for chunk in chunks:
        # 每个 chunk 长度不超过 chunk_size
        assert len(chunk.page_content) <= config.CHUNK_SIZE
        # 保留来源和页码
        assert chunk.metadata.get("source") == "test.pdf"
        assert chunk.metadata.get("page") == 0


def test_split_documents_preserves_overlap(sample_documents):
    """验证 chunk 之间存在 overlap（重叠）。"""
    chunks = split_documents(sample_documents)
    if len(chunks) >= 2:
        first_end = chunks[0].page_content[-config.CHUNK_OVERLAP:]
        second_start = chunks[1].page_content[:config.CHUNK_OVERLAP]
        assert first_end == second_start


# ============================================================
# PDF 去重测试
# ============================================================

def test_compute_file_hash(tmp_path):
    """验证文件哈希计算：相同内容哈希相同，不同内容哈希不同。"""
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_text("hello world")
    f2.write_text("hello world")

    assert compute_file_hash(f1) == compute_file_hash(f2)

    f2.write_text("different content")
    assert compute_file_hash(f1) != compute_file_hash(f2)


# ============================================================
# PDF 加载测试（错误处理）
# ============================================================

def test_load_pdf_nonexistent():
    """验证加载不存在的文件抛出清晰错误。"""
    from src.loader import load_pdf

    with pytest.raises(ValueError, match="文件不存在"):
        load_pdf("D:/不存在的文件.pdf")


def test_load_pdf_wrong_extension(tmp_path):
    """验证加载非 PDF 文件抛出错误。"""
    from src.loader import load_pdf

    fake = tmp_path / "not_pdf.txt"
    fake.write_text("hello")
    with pytest.raises(ValueError, match="不是 PDF"):
        load_pdf(fake)


# ============================================================
# 检索分发测试
# ============================================================

def _fake_docs():
    from langchain_core.documents import Document

    return [
        Document(page_content="片段A", metadata={"source": "a.pdf", "page": 1, "start_index": 0}),
        Document(page_content="片段B", metadata={"source": "b.pdf", "page": 2, "start_index": 0}),
    ]


def test_search_vector_mode_delegates_to_vector_search(monkeypatch):
    """vector 模式应直接走向量检索，并带上分数占位。"""
    from src import retriever

    captured = {}

    def fake_vector_search(query, top_k):
        captured["query"] = query
        captured["top_k"] = top_k
        return _fake_docs()

    monkeypatch.setattr(retriever, "vector_search", fake_vector_search)

    results = retriever.search("测试问题", top_k=2, mode="vector")
    assert len(results) == 2
    assert results[0][0].metadata["source"] == "a.pdf"
    assert captured == {"query": "测试问题", "top_k": 2}


def test_search_bm25_mode_delegates_to_index(monkeypatch):
    """bm25 模式应走 BM25 索引，并保留其返回的分数。"""
    from src import retriever

    class FakeIndex:
        def search(self, query, top_k):
            return [(doc, 1.23) for doc in _fake_docs()[:1]]

    monkeypatch.setattr(retriever, "get_bm25_index", lambda force_reload=False: FakeIndex())

    results = retriever.search("测试问题", top_k=3, mode="bm25")
    assert len(results) == 1
    assert results[0][1] == pytest.approx(1.23)


def test_search_hybrid_fuses_both_sources(monkeypatch):
    """hybrid 模式应同时调用两路检索，并用 RRF 融合去重。"""
    from src import retriever

    docs = _fake_docs()

    class FakeIndex:
        def search(self, query, top_k):
            return [(docs[0], 2.0)]

    monkeypatch.setattr(retriever, "vector_search", lambda q, k: docs)
    monkeypatch.setattr(retriever, "get_bm25_index", lambda force_reload=False: FakeIndex())

    results = retriever.search("测试问题", top_k=5, mode="hybrid")
    contents = [doc.page_content for doc, _score in results]
    # 两路都出现的 docs[0] 应排在最前，且结果不重复
    assert contents[0] == "片段A"
    assert len(contents) == len(set(contents))


def test_retrieve_returns_documents_only(monkeypatch):
    """retrieve() 是 search() 的简化封装，只返回文本块。"""
    from src import retriever

    monkeypatch.setattr(retriever, "vector_search", lambda q, k: _fake_docs())
    results = retriever.retrieve("测试问题", top_k=2)
    assert all(hasattr(doc, "page_content") for doc in results)


def test_ask_empty_question(monkeypatch):
    """验证空问题抛出错误（在检索之前就该被拦下）。"""
    from src import rag_chain

    with pytest.raises(ValueError, match="问题不能为空"):
        rag_chain.ask("   ")
