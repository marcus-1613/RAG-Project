"""检索质量评估模块。

为什么必须做评估：
    "加了混合检索和 rerank，效果应该更好"——这是个**感觉**，不是结论。
    没有量化指标，就无法回答面试官那句"你怎么知道改进了？"，
    也无法判断某个配置在你的语料上究竟有没有用。

三个指标：
    - **Hit@K**：前 K 条里至少命中一条相关片段的提问占比。回答"找得到吗"。
    - **Recall@K**：前 K 条覆盖了多少比例的全部相关片段。回答"找得全吗"。
      （与 Hit@K 的区别：Hit 只看有没有，Recall 看覆盖了多少）
    - **MRR**：第一条相关片段排名的倒数取平均。回答"排得靠前吗"。

相关性判定（本模块的假设，必须明确）：
    评估集为每个问题标注若干**关键词**；某个文本块只要包含其中任意一个
    关键词，就视为该问题的相关片段。

    这是一个**粗略但可复现**的判定方式：不需要人工逐条标注相关性等级，
    换语料也能自动重算。代价是它偏向"字面命中"，对纯语义改写的问题
    会低估向量检索的表现。

    真实项目中应当用人工标注的相关性标签，或引入 LLM 作为评判员。
    本模块的定位是**让评估流程可运行、可对比**，而不是给出权威分数。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from langchain_core.documents import Document

from src.fusion import doc_key


def normalize_for_match(text: str) -> str:
    """去掉全部空白字符，用于关键词匹配。

    为什么必须做这一步：
        PDF 提取出来的文本带有**排版换行**，而换行并不遵守词的边界。
        例如原文里的 "Transformer" 在 PDF 中被行宽切成两行后，
        PyPDFLoader 提取出来的实际是 `"Transfo\\nrmer"`——直接做子串匹配
        永远找不到 "Transformer"，评估就会把一个语料里明明存在的答案
        判成"不存在"。

        去掉空白后 `"Transfo\\nrmer"` 重新变成 `"Transformer"`，匹配恢复正常。
        这也顺带解决了中文被换行切开（"一\\n个重要方向"）的情况。

    注意：这里只做"比较用"的归一化，不修改存进向量库的原始文本——
    检索时仍然需要保留原始排版信息。

    Args:
        text: 任意文本。

    Returns:
        str: 去除所有空白字符后的文本。
    """
    return "".join(text.split())


@dataclass
class EvalQuestion:
    """一条评估样本。"""

    question: str
    keywords: list[str]
    category: str = "general"


@dataclass
class QueryOutcome:
    """单个问题的评估结果。"""

    question: str
    category: str
    ground_truth_count: int
    retrieved_count: int
    first_hit_rank: int | None

    @property
    def hit(self) -> bool:
        """前 K 条中是否命中至少一条相关片段。"""
        return self.first_hit_rank is not None


@dataclass
class EvalReport:
    """一次评估的汇总结果。"""

    mode: str
    k: int
    questions: int
    hit_rate: float
    recall: float
    mrr: float
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    outcomes: list[QueryOutcome] = field(default_factory=list)

    def as_row(self) -> str:
        """格式化为对比表格的一行。"""
        return (
            f"{self.mode:<16} {self.hit_rate:>8.3f} {self.recall:>9.3f} "
            f"{self.mrr:>7.3f}"
        )


def load_eval_set(path: str | Path) -> list[EvalQuestion]:
    """从 JSONL 文件读取评估集。

    每行一个 JSON 对象：{"question": "...", "keywords": ["..."], "category": "..."}
    以 # 开头的行和空行会被忽略，方便在文件里写注释。

    Args:
        path: JSONL 文件路径。

    Returns:
        list[EvalQuestion]: 解析后的评估样本。

    Raises:
        ValueError: 文件缺失，或某行缺少 question / keywords 字段时抛出。
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"评估集文件不存在：{path}")

    questions: list[EvalQuestion] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} 第 {lineno} 行不是合法 JSON：{exc}") from exc

        if not item.get("question") or not item.get("keywords"):
            raise ValueError(
                f"{path.name} 第 {lineno} 行缺少 question 或 keywords 字段"
            )
        questions.append(
            EvalQuestion(
                question=str(item["question"]),
                keywords=[str(k) for k in item["keywords"]],
                category=str(item.get("category", "general")),
            )
        )
    return questions


def build_ground_truth(
    corpus: Iterable[Document], keywords: list[str]
) -> set[str]:
    """在整份语料中找出所有与关键词相关的文本块。

    匹配前会对正文和关键词同时做空白归一化，以抵抗 PDF 换行带来的
    断词问题（详见 normalize_for_match 的说明）。

    Args:
        corpus: 语料中的全部文本块。
        keywords: 该问题的相关关键词。

    Returns:
        set[str]: 相关文本块的 doc_key 集合（即 Recall 的分母）。
    """
    normalized_keywords = [
        normalize_for_match(keyword)
        for keyword in keywords
        if normalize_for_match(keyword)
    ]
    truth: set[str] = set()
    for doc in corpus:
        content = normalize_for_match(doc.page_content)
        if any(keyword in content for keyword in normalized_keywords):
            truth.add(doc_key(doc))
    return truth


def evaluate(
    questions: list[EvalQuestion],
    corpus: list[Document],
    retrieve_fn: Callable[[str], list[Document]],
    k: int = 5,
    mode: str = "default",
) -> EvalReport:
    """对一种检索配置跑完整评估。

    Args:
        questions: 评估样本。
        corpus: 语料中的全部文本块（用于构建相关性真值）。
        retrieve_fn: 检索函数，接收问题、返回已排序的文本块列表。
        k: 截断位置，只评估前 K 条。
        mode: 该配置的名称，用于报告展示。

    Returns:
        EvalReport: 汇总指标 + 逐题明细。
    """
    outcomes: list[QueryOutcome] = []
    hit_total = 0
    recall_total = 0.0
    mrr_total = 0.0

    for item in questions:
        ground_truth = build_ground_truth(corpus, item.keywords)
        retrieved = [doc_key(doc) for doc in retrieve_fn(item.question)[:k]]

        first_hit_rank: int | None = None
        for rank, key in enumerate(retrieved, start=1):
            if key in ground_truth:
                first_hit_rank = rank
                break

        hit_count = len(set(retrieved) & ground_truth)
        if first_hit_rank is not None:
            hit_total += 1
            mrr_total += 1.0 / first_hit_rank
        if ground_truth:
            recall_total += hit_count / len(ground_truth)

        outcomes.append(
            QueryOutcome(
                question=item.question,
                category=item.category,
                ground_truth_count=len(ground_truth),
                retrieved_count=len(retrieved),
                first_hit_rank=first_hit_rank,
            )
        )

    total = len(outcomes)
    report = EvalReport(
        mode=mode,
        k=k,
        questions=total,
        hit_rate=hit_total / total if total else 0.0,
        recall=recall_total / total if total else 0.0,
        mrr=mrr_total / total if total else 0.0,
        outcomes=outcomes,
    )
    report.by_category = _summarize_by_category(outcomes)
    return report


def _summarize_by_category(
    outcomes: list[QueryOutcome],
) -> dict[str, dict[str, float]]:
    """按问题类别分组统计，用于定位"哪种检索在什么场景更强"。"""
    buckets: dict[str, list[QueryOutcome]] = {}
    for outcome in outcomes:
        buckets.setdefault(outcome.category, []).append(outcome)

    summary: dict[str, dict[str, float]] = {}
    for category, items in buckets.items():
        count = len(items)
        hit = sum(1 for i in items if i.hit)
        mrr = sum(
            (1.0 / i.first_hit_rank) for i in items if i.first_hit_rank is not None
        )
        summary[category] = {
            "questions": float(count),
            "hit_rate": hit / count if count else 0.0,
            "mrr": mrr / count if count else 0.0,
        }
    return summary


def format_comparison(reports: list[EvalReport]) -> str:
    """把多份报告格式化成一张对比表。

    Args:
        reports: 各检索配置的评估报告。

    Returns:
        str: 可直接打印的对齐文本。
    """
    lines = [
        f"{'检索模式':<12} {'Hit@K':>8} {'Recall@K':>9} {'MRR':>7}",
        "-" * 44,
    ]
    lines.extend(report.as_row() for report in reports)

    categories = sorted(
        {c for report in reports for c in report.by_category}
    )
    if categories:
        lines.append("")
        lines.append("按问题类型拆分（Hit@K）：")
        header = f"{'检索模式':<12}" + "".join(f"{c:>14}" for c in categories)
        lines.append(header)
        lines.append("-" * len(header))
        for report in reports:
            row = f"{report.mode:<12}"
            for category in categories:
                value = report.by_category.get(category, {}).get("hit_rate")
                row += f"{value:>14.3f}" if value is not None else f"{'-':>14}"
            lines.append(row)

    return "\n".join(lines)
