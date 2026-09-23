"""RAG 业务编排模块。

将「检索 → 构造 Prompt → 调用 LLM → 返回答案」串成一个完整流程，
对外只暴露一个简洁的 ask 函数，app.py 无需关心内部实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document

from src import config
from src.llm import get_llm
from src.retriever import retrieve


# 中文 RAG Prompt 模板
_SYSTEM_PROMPT = (
    "你是一个知识库问答助手。\n"
    "请严格根据下面提供的「知识库上下文」回答问题。\n\n"
    "要求：\n"
    "1. 优先根据知识库上下文回答，不要编造上下文之外的信息。\n"
    "2. 如果上下文信息不足，请明确回答："
    "「知识库中没有找到足够的信息来回答这个问题。」\n"
    "3. 回答要清晰、简洁。\n"
    "4. 如果可能，指出答案来自哪个来源。\n"
    "5. 不要把检索到的文本机械复制成一大段，用自己的话概括。\n"
    "6. 区分「知识库事实」和「模型推测」：基于上下文的回答是事实，"
    "超出上下文的推断请明确标注为推测。"
)


@dataclass
class RAGResult:
    """RAG 问答的返回结果。"""

    answer: str
    retrieved_documents: list[Document] = field(default_factory=list)


def _build_context(documents: list[Document]) -> str:
    """将检索到的文本块拼接为上下文字符串。

    每个文本块标注编号、来源文件名和页码，方便模型和用户定位。

    Args:
        documents: 检索到的文本块列表。

    Returns:
        str: 拼接好的上下文字符串。
    """
    parts: list[str] = []
    for idx, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "未知页")
        content = doc.page_content.strip()
        parts.append(
            f"[片段 {idx}] 来源：{source}，页码：{page}\n{content}"
        )
    return "\n\n".join(parts)


def _build_user_prompt(context: str, question: str) -> str:
    """构造发送给 LLM 的完整用户 Prompt。

    Args:
        context: 拼接好的知识库上下文。
        question: 用户问题。

    Returns:
        str: 完整的用户 Prompt。
    """
    return (
        f"知识库上下文：\n{context}\n\n"
        f"用户问题：{question}\n\n"
        "请根据上述上下文回答。"
    )


def ask(question: str, top_k: int | None = None) -> RAGResult:
    """执行一次完整的 RAG 问答。

    流程：检索 Top-K 文本块 → 构造 Prompt → 调用 DeepSeek → 返回答案。

    Args:
        question: 用户问题。
        top_k: 检索数量，默认 config.TOP_K。

    Returns:
        RAGResult: 包含 answer 和 retrieved_documents 的结果对象。

    Raises:
        ValueError: 问题为空、知识库为空或检索不到结果时抛出。
        Exception: LLM 调用失败时抛出（由上层转换为用户可读信息）。
    """
    question = question.strip()
    if not question:
        raise ValueError("问题不能为空。")

    # 1. 检索
    documents = retrieve(question, top_k=top_k)
    if not documents:
        raise ValueError(
            "知识库为空，或没有检索到相关内容。\n"
            "请先上传 PDF 并点击「建立/更新知识库」。"
        )

    # 2. 构造 Prompt
    context = _build_context(documents)
    user_prompt = _build_user_prompt(context, question)

    # 3. 调用 LLM
    llm = get_llm()
    response = llm.invoke(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    answer = response.content.strip() if hasattr(response, "content") else str(response)

    return RAGResult(answer=answer, retrieved_documents=documents)


# 供测试或命令行直接调用
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法：python -m src.rag_chain \"你的问题\"")
        sys.exit(1)

    question = sys.argv[1]
    result = ask(question)
    print("=" * 60)
    print("回答：")
    print(result.answer)
    print("=" * 60)
    print(f"检索到 {len(result.retrieved_documents)} 个文本块：")
    for i, doc in enumerate(result.retrieved_documents, 1):
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "未知")
        print(f"\n[{i}] {source} (页码 {page})")
        print(doc.page_content[:100])
