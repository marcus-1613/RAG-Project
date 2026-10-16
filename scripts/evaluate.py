"""检索质量评估脚本：横向对比不同检索模式。

用法：
    # 先建库（只需一次）
    python scripts/verify_rag.py --build

    # 对比全部四种模式
    python scripts/evaluate.py

    # 只对比指定模式
    python scripts/evaluate.py --modes vector,hybrid

    # 打印逐题明细，定位是哪些问题拖低了指标
    python scripts/evaluate.py --verbose

说明：
    - 评估集默认读取 eval/qa_set.jsonl，可用 --set 指定其他文件。
    - 整个评估过程**不需要 DeepSeek API Key**：只测检索，不调生成。
    - hybrid_rerank 模式需要 Cross-Encoder 模型；若模型不可用，
      会自动降级为融合排序，脚本会给出提示。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows 控制台默认 GBK，输出 emoji 会抛 UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.evaluate import (  # noqa: E402
    EvalReport,
    evaluate,
    format_comparison,
    load_eval_set,
)
from src.retriever import clear_retrieval_cache, search  # noqa: E402
from src.vector_store import get_all_documents  # noqa: E402


def make_retrieve_fn(mode: str):
    """构造某个检索模式的检索函数（供 evaluate 调用）。"""

    def _retrieve(question: str):
        return [doc for doc, _score in search(question, mode=mode)]

    return _retrieve


def print_details(report: EvalReport) -> None:
    """打印某个报告下每条问题的命中情况。"""
    print(f"\n【{report.mode}】逐题明细：")
    for outcome in report.outcomes:
        if outcome.first_hit_rank is None:
            mark = "未命中"
        else:
            mark = f"命中第 {outcome.first_hit_rank} 位"
        print(
            f"  [{mark:<12}] {outcome.question}"
            f"   (相关片段共 {outcome.ground_truth_count} 条)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="检索质量评估脚本")
    parser.add_argument(
        "--modes",
        default="vector,bm25,hybrid,hybrid_rerank",
        help="要对比的检索模式，逗号分隔（默认全部四种）",
    )
    parser.add_argument("--k", type=int, default=config.TOP_K, help="截断位置 K")
    parser.add_argument("--set", dest="eval_set", default=None, help="评估集 JSONL 路径")
    parser.add_argument("--verbose", action="store_true", help="打印逐题明细")
    args = parser.parse_args()

    # 1. 读取语料
    corpus = get_all_documents()
    if not corpus:
        print("知识库为空。请先执行：python scripts/verify_rag.py --build")
        sys.exit(1)
    print(f"语料规模：{len(corpus)} 个文本块")

    # 2. 读取评估集
    eval_path = Path(args.eval_set) if args.eval_set else config.EVAL_SET_PATH
    questions = load_eval_set(eval_path)
    print(f"评估集：{eval_path}")
    print(f"问题数量：{len(questions)}")
    print(f"截断位置：K = {args.k}")

    # 语料已就绪，确保 BM25 索引是最新的
    clear_retrieval_cache()

    # 3. 逐个模式评估
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    reports: list[EvalReport] = []
    for mode in modes:
        try:
            config.ensure_valid_mode(mode)
        except ValueError as exc:
            print(f"\n跳过无效模式：{exc}")
            continue
        print(f"\n正在评估 {mode} ...")
        reports.append(
            evaluate(
                questions=questions,
                corpus=corpus,
                retrieve_fn=make_retrieve_fn(mode),
                k=args.k,
                mode=mode,
            )
        )

    if not reports:
        print("没有可用的评估结果。")
        sys.exit(1)

    # 4. 输出对比
    print("\n" + "=" * 60)
    print("检索质量对比")
    print("=" * 60)
    print(format_comparison(reports))

    print(
        "\n指标含义：Hit@K = 前 K 条至少命中一条相关片段的提问占比；"
        "\n          Recall@K = 前 K 条覆盖的相关片段比例；"
        "\n          MRR = 第一条相关片段排名的倒数均值（越高说明排得越靠前）。"
    )

    if args.verbose:
        for report in reports:
            print_details(report)


if __name__ == "__main__":
    main()
