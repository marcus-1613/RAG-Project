"""项目集中配置模块。

所有环境变量、模型参数、路径、切分参数等配置统一在此处定义，
其他模块一律从这里读取，避免配置散落在多个文件。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录下的 .env 文件
_BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_BASE_DIR / ".env")


# ============================================================
# 路径配置
# ============================================================

# 项目根目录
BASE_DIR = _BASE_DIR

# PDF 文件目录
PDF_DIR = BASE_DIR / "data" / "pdfs"

# ChromaDB 持久化目录
CHROMA_DIR = BASE_DIR / "data" / "chroma"

# 评估集文件
EVAL_SET_PATH = BASE_DIR / "eval" / "qa_set.jsonl"


# ============================================================
# DeepSeek 配置
# ============================================================

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


# ============================================================
# Embedding 配置
# ============================================================

# 本地 HuggingFace 中文语义检索模型
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")


# ============================================================
# 文本切分配置
# ============================================================

# 每个文本块的最大字符数
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))

# 相邻文本块之间的重叠字符数
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# 文本块之间的分隔符优先级（从强到弱）
CHUNK_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


# ============================================================
# 检索配置
# ============================================================

# 最终返回给 LLM 的文本块数量
TOP_K = int(os.getenv("TOP_K", "5"))

# 检索模式：
#   vector        —— 仅向量检索（原始行为，作为基线）
#   bm25          —— 仅关键词检索
#   hybrid        —— 向量 + BM25，用 RRF 融合
#   hybrid_rerank —— 在 hybrid 之上再做 Cross-Encoder 精排
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").strip().lower()

VALID_RETRIEVAL_MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")

# 每路检索各自召回的候选数量（融合前）。需要大于 TOP_K，
# 否则融合没有腾挪空间——它只能重排已有的候选。
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "20"))

# RRF 融合的平滑常数
RRF_K = int(os.getenv("RRF_K", "60"))

# 是否启用 Cross-Encoder 精排（仅 hybrid_rerank 模式下生效）。
# 关闭后 hybrid_rerank 等同于 hybrid，便于对比精排带来的增益。
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Cross-Encoder 重排模型
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")


# ============================================================
# 校验函数
# ============================================================

def ensure_api_key() -> str:
    """校验 DeepSeek API Key 是否已配置，缺失时抛出清晰的中文错误。

    Returns:
        str: 已配置的 API Key。

    Raises:
        ValueError: API Key 未配置时抛出。
    """
    if not DEEPSEEK_API_KEY:
        raise ValueError(
            "未检测到 DeepSeek API Key。\n"
            "请复制 .env.example 为 .env，并在其中填入你的 "
            "DEEPSEEK_API_KEY（可在 https://platform.deepseek.com 申请）。"
        )
    return DEEPSEEK_API_KEY


def ensure_valid_mode(mode: str) -> str:
    """校验检索模式取值是否合法。

    配置写错时**立即报错**，而不是静默退回默认值——静默回退会让人
    以为在跑 hybrid，实际跑的是 vector，评估结果就全错了。

    Args:
        mode: 待校验的模式名。

    Returns:
        str: 校验通过的模式名。

    Raises:
        ValueError: 模式名不在 VALID_RETRIEVAL_MODES 中时抛出。
    """
    mode = (mode or "").strip().lower()
    if mode not in VALID_RETRIEVAL_MODES:
        raise ValueError(
            f"无效的检索模式：{mode!r}\n"
            f"可选值：{' / '.join(VALID_RETRIEVAL_MODES)}"
        )
    return mode
