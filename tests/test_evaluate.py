"""检索评估模块的单元测试（纯逻辑，使用假的检索函数）。"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.evaluate import (
    EvalQuestion,
    build_ground_truth,
    evaluate,
    format_comparison,
    load_eval_set,
    normalize_for_match,
)


def _doc(text: str, start: int) -> Document:
    return Document(
        page_content=text,
        metadata={"source": "a.pdf", "page": 0, "start_index": start},
    )


CORPUS = [
    _doc("RAG 是检索增强生成技术。", 0),
    _doc("ChromaDB 是向量数据库。", 1),
    _doc("光合作用发生在叶绿体中。", 2),
]


# ============================================================
# 评估集读取
# ============================================================

def test_load_eval_set_parses_and_skips_comments(tmp_path):
    path = tmp_path / "set.jsonl"
    path.write_text(
        "# 这是注释\n"
        "\n"
        '{"question": "q1", "keywords": ["k1"], "category": "keyword"}\n'
        '{"question": "q2", "keywords": ["k2"]}\n',
        encoding="utf-8",
    )
    items = load_eval_set(path)
    assert len(items) == 2
    assert items[0].category == "keyword"
    assert items[1].category == "general"  # 缺省类别


def test_load_eval_set_missing_field_raises(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"question": "只有问题没有关键词"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="缺少"):
        load_eval_set(path)


def test_load_eval_set_missing_file_raises():
    with pytest.raises(ValueError, match="不存在"):
        load_eval_set("D:/绝不存在/qa_set.jsonl")


# ============================================================
# 相关性真值
# ============================================================

def test_build_ground_truth_matches_keywords():
    truth = build_ground_truth(CORPUS, ["ChromaDB"])
    assert truth == {"a.pdf#0#1"}


def test_build_ground_truth_returns_empty_when_no_match():
    assert build_ground_truth(CORPUS, ["完全不存在的词"]) == set()


def test_normalize_for_match_strips_all_whitespace():
    assert normalize_for_match("Transfo\nrmer 架构") == "Transformer架构"
    assert normalize_for_match("一\n个  重要\t方向") == "一个重要方向"


def test_ground_truth_survives_pdf_line_wrapping():
    """PDF 把英文单词从行中断开时，关键词匹配仍应命中。

    这不是假想的场景：PyPDFLoader 提取长英文单词时，实际拿到的文本
    就是 "Transfo\\nrmer"。不做空白归一化，这一题的相关片段会被判成 0 条，
    对所有检索模式都变成必挂——评估结果就失真了。
    """
    wrapped = [
        Document(
            page_content="循环神经网络适合处理序列数据，而 Transfo\nrmer 架构取得了成功。",
            metadata={"source": "a.pdf", "page": 0, "start_index": 0},
        )
    ]
    assert build_ground_truth(wrapped, ["Transformer"]) == {"a.pdf#0#0"}


# ============================================================
# 指标计算
# ============================================================

def test_evaluate_perfect_retriever():
    """每次都把相关片段排在第一，各项指标应为满分。"""
    questions = [EvalQuestion(question="q", keywords=["ChromaDB"])]

    report = evaluate(
        questions,
        CORPUS,
        retrieve_fn=lambda _q: [CORPUS[1], CORPUS[0]],
        k=5,
        mode="perfect",
    )
    assert report.hit_rate == 1.0
    assert report.mrr == 1.0
    assert report.recall == 1.0


def test_evaluate_counts_rank_for_mrr():
    """相关片段排在第 2 位时，MRR 应为 1/2。"""
    questions = [EvalQuestion(question="q", keywords=["ChromaDB"])]
    report = evaluate(
        questions,
        CORPUS,
        retrieve_fn=lambda _q: [CORPUS[0], CORPUS[1]],
        k=5,
        mode="second",
    )
    assert report.hit_rate == 1.0
    assert report.mrr == pytest.approx(0.5)


def test_evaluate_scores_zero_when_nothing_relevant_retrieved():
    questions = [EvalQuestion(question="q", keywords=["ChromaDB"])]
    report = evaluate(
        questions,
        CORPUS,
        retrieve_fn=lambda _q: [CORPUS[2]],
        k=5,
        mode="miss",
    )
    assert report.hit_rate == 0.0
    assert report.mrr == 0.0


def test_evaluate_respects_k_cutoff():
    """相关片段排在第 3 位，K=2 时应判定为未命中。"""
    questions = [EvalQuestion(question="q", keywords=["ChromaDB"])]
    report = evaluate(
        questions,
        CORPUS,
        retrieve_fn=lambda _q: [CORPUS[0], CORPUS[2], CORPUS[1]],
        k=2,
        mode="cut",
    )
    assert report.hit_rate == 0.0


def test_evaluate_by_category_breakdown():
    questions = [
        EvalQuestion(question="q1", keywords=["ChromaDB"], category="keyword"),
        EvalQuestion(question="q2", keywords=["不存在词"], category="semantic"),
    ]
    report = evaluate(
        questions,
        CORPUS,
        retrieve_fn=lambda q: [CORPUS[1]] if q == "q1" else [CORPUS[2]],
        k=5,
        mode="mixed",
    )
    assert report.by_category["keyword"]["hit_rate"] == 1.0
    assert report.by_category["semantic"]["hit_rate"] == 0.0


def test_format_comparison_contains_all_modes():
    questions = [EvalQuestion(question="q", keywords=["ChromaDB"])]
    reports = [
        evaluate(questions, CORPUS, lambda _q: [CORPUS[1]], k=5, mode=m)
        for m in ("vector", "hybrid")
    ]
    text = format_comparison(reports)
    assert "vector" in text
    assert "hybrid" in text
    assert "Hit@K" in text
