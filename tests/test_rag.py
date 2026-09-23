"""RAG 核心流程测试。

覆盖：配置读取、文本切分、PDF 去重、以及检索流程的基本逻辑。
需要真实 API / 模型的步骤（embedding、LLM 调用）使用 mock，
保证没有 API Key 也能运行全部测试。
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
# 检索流程测试（mock embedding）
# ============================================================

def test_retrieve_flow(monkeypatch, tmp_path):
    """验证检索流程的基本逻辑（mock 掉 embedding 和 Chroma）。

    这里不测试真实的 ChromaDB，而是验证 retriever 的调用逻辑，
    避免测试依赖模型下载和真实向量库。
    """
    from langchain_core.documents import Document
    from src import retriever

    fake_docs = [
        Document(page_content="片段A", metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="片段B", metadata={"source": "b.pdf", "page": 2}),
    ]

    class FakeRetriever:
        def invoke(self, query):
            return fake_docs

    monkeypatch.setattr(retriever, "get_retriever", lambda top_k=None: FakeRetriever())

    results = retriever.retrieve("测试问题")
    assert len(results) == 2
    assert results[0].metadata["source"] == "a.pdf"
    assert results[0].metadata["page"] == 1


def test_ask_empty_question(monkeypatch):
    """验证空问题抛出错误。"""
    from src import rag_chain

    with pytest.raises(ValueError, match="问题不能为空"):
        rag_chain.ask("   ")
