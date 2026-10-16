"""Rerank 模块的单元测试。

重点是验证**降级路径**：模型不可用时必须原样返回候选，
而不是抛异常阻断整个问答流程。真实的 Cross-Encoder 精排效果
需要下载模型才能测，这里用假模型验证接口与排序逻辑。
"""

from __future__ import annotations

from langchain_core.documents import Document

from src.rerank import Reranker


def _docs() -> list[Document]:
    return [
        Document(page_content=f"片段{i}", metadata={"source": "a.pdf", "start_index": i})
        for i in range(3)
    ]


def test_rerank_degrades_gracefully_without_model(monkeypatch):
    """模型加载失败时应原样返回，并保持顺序不变。"""
    reranker = Reranker("不存在的模型")
    monkeypatch.setattr(reranker, "_ensure_model", lambda: None)

    docs = _docs()
    result = reranker.rerank("问题", docs, top_n=2)

    assert [doc.page_content for doc, _score in result] == ["片段0", "片段1"]
    assert all(score == 0.0 for _doc, score in result)


def test_rerank_returns_empty_for_empty_candidates():
    reranker = Reranker("任意模型")
    assert reranker.rerank("问题", [], top_n=5) == []


def test_rerank_reorders_using_model_scores(monkeypatch):
    """按模型打分重排：分数高的排前面。"""

    class FakeModel:
        def predict(self, pairs):
            # 故意给第一个候选最低分、第三个最高分
            return [0.1, 0.5, 0.9]

    reranker = Reranker("假模型")
    monkeypatch.setattr(reranker, "_ensure_model", lambda: FakeModel())

    result = reranker.rerank("问题", _docs(), top_n=3)
    assert [doc.page_content for doc, _score in result] == ["片段2", "片段1", "片段0"]


def test_rerank_predict_failure_falls_back(monkeypatch):
    """预测阶段抛异常时同样降级，不影响主流程。"""

    class BrokenModel:
        def predict(self, pairs):
            raise RuntimeError("模拟推理失败")

    reranker = Reranker("假模型")
    monkeypatch.setattr(reranker, "_ensure_model", lambda: BrokenModel())

    result = reranker.rerank("问题", _docs(), top_n=2)
    assert [doc.page_content for doc, _score in result] == ["片段0", "片段1"]


def test_available_property_reflects_load_failure():
    reranker = Reranker("模型")
    assert reranker.available is True
    reranker._load_failed = True
    assert reranker.available is False
