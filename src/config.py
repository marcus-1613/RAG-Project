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

# 默认返回的最相似文本块数量
TOP_K = int(os.getenv("TOP_K", "5"))


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
