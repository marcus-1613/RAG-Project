"""Embedding 模块。

使用本地 HuggingFace 模型（默认 BAAI/bge-small-zh-v1.5）生成文本向量。
选择本地模型的原因：DeepSeek 目前不提供可直接使用的 embedding 接口，
本地模型无需额外的 API Key，且对中文语义检索效果良好。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src import config


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """获取（并缓存）Embedding 模型实例。

    使用 lru_cache 确保整个程序只加载一次模型，避免重复占用内存。

    Returns:
        HuggingFaceEmbeddings: 可调用的 embedding 对象，
            拥有 embed_documents 和 embed_query 方法。

    Raises:
        RuntimeError: 模型加载失败时抛出（如网络问题导致模型无法下载）。
    """
    try:
        return HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Embedding 模型加载失败：{config.EMBEDDING_MODEL}\n"
            f"原因：{exc}\n"
            "提示：首次运行需要联网下载模型；若下载缓慢，可在 .env 中设置 "
            "HF_ENDPOINT=https://hf-mirror.com 使用国内镜像。"
        ) from exc
