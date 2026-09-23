"""PDF 加载模块。

使用 LangChain 的 PyPDFLoader 将 PDF 文件加载为 Document 列表，
并保留文件名和页码等 metadata，便于后续检索时展示来源信息。
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(pdf_path: str | Path) -> list[Document]:
    """加载单个 PDF 文件为 Document 列表。

    Args:
        pdf_path: PDF 文件的路径。

    Returns:
        list[Document]: 每个 Document 对应 PDF 中的一页，
            其 metadata 中包含 source（文件名）和 page（页码）。

    Raises:
        ValueError: 文件不存在、不是 PDF，或 PDF 无法读取/没有有效文本时抛出。
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise ValueError(f"文件不存在：{pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"不是 PDF 文件：{pdf_path.name}")

    try:
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
    except Exception as exc:  # noqa: BLE001 - 转化为用户可读错误
        raise ValueError(f"读取 PDF 失败：{pdf_path.name}（{exc}）") from exc

    if not documents:
        raise ValueError(f"PDF 内容为空：{pdf_path.name}")

    # 过滤掉没有有效文本的页，并为每页补全来源信息
    valid_documents: list[Document] = []
    for doc in documents:
        text = doc.page_content.strip()
        if not text:
            continue
        # 确保 metadata 里有文件名和页码
        doc.metadata["source"] = doc.metadata.get("source", pdf_path.name)
        doc.metadata["page"] = doc.metadata.get("page", 0)
        valid_documents.append(doc)

    if not valid_documents:
        raise ValueError(
            f"PDF 中没有提取到有效文本：{pdf_path.name}\n"
            "（可能是纯扫描图片版 PDF，目前暂不支持）"
        )

    return valid_documents


def load_pdf_directory(pdf_dir: str | Path) -> list[Document]:
    """加载目录下所有 PDF 文件，返回合并后的 Document 列表。

    Args:
        pdf_dir: PDF 文件所在目录。

    Returns:
        list[Document]: 目录下所有 PDF 的 Document 列表。
    """
    pdf_dir = Path(pdf_dir)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    all_documents: list[Document] = []
    for pdf_file in pdf_files:
        all_documents.extend(load_pdf(pdf_file))
    return all_documents
