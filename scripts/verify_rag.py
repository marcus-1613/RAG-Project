"""命令行端到端验证脚本。

用途：在无 Streamlit 界面的情况下，验证 RAG 全链路是否跑通。

用法：
    # 1. 只建库（不需要 API Key）
    python scripts/verify_rag.py --build

    # 2. 建库 + 检索（不需要 API Key）
    python scripts/verify_rag.py --build --query "什么是RAG"

    # 3. 建库 + 检索 + 调用 DeepSeek 生成回答（需要 API Key）
    python scripts/verify_rag.py --build --query "什么是RAG" --answer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Windows 控制台默认使用 GBK 编码，无法输出 ✅ 等 emoji，
# 会在 print 时抛出 UnicodeEncodeError。这里统一转为 UTF-8 避免崩溃。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.loader import load_pdf_directory  # noqa: E402
from src.splitter import split_documents  # noqa: E402
from src.vector_store import add_documents  # noqa: E402
from src.retriever import retrieve  # noqa: E402


def build_knowledge_base() -> None:
    """将 data/pdfs 下的所有 PDF 加载、切分并写入 ChromaDB。"""
    print("[1/3] 正在读取 PDF ...")
    documents = load_pdf_directory(config.PDF_DIR)
    print(f"      读取到 {len(documents)} 页")

    print("[2/3] 正在切分文本 ...")
    chunks = split_documents(documents)
    print(f"      切分为 {len(chunks)} 个文本块")

    print("[3/3] 正在生成 Embedding 并写入 ChromaDB ...")
    add_documents(chunks)
    print("      入库完成 ✅")

    print(f"\n知识库位置：{config.CHROMA_DIR}")


def do_query(question: str) -> None:
    """检索并打印 Top-K 结果。"""
    print(f"\n问题：{question}")
    print("正在检索 ...")
    results = retrieve(question)
    print(f"检索到 {len(results)} 个相关片段：\n")
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "未知")
        print(f"[{i}] 来源：{source} ｜ 页码：{page}")
        print(f"    {doc.page_content[:80]}...\n")


def do_answer(question: str) -> None:
    """调用 DeepSeek 生成最终回答。"""
    from src.rag_chain import ask

    print("\n正在调用 DeepSeek 生成回答 ...")
    result = ask(question)
    print("\n" + "=" * 60)
    print("最终回答：")
    print(result.answer)
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 全链路验证脚本")
    parser.add_argument("--build", action="store_true", help="建立/更新知识库")
    parser.add_argument("--query", type=str, help="检索问题")
    parser.add_argument("--answer", action="store_true", help="调用 DeepSeek 生成回答")
    args = parser.parse_args()

    if args.build:
        build_knowledge_base()

    if args.query:
        do_query(args.query)

    if args.answer:
        if not args.query:
            print("错误：--answer 需要配合 --query 使用")
            sys.exit(1)
        do_answer(args.query)


if __name__ == "__main__":
    main()
