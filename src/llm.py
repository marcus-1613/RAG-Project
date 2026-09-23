"""LLM 模块。

初始化 DeepSeek Chat 模型。API Key 只从环境变量读取，绝不硬编码。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from src import config


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """获取（并缓存）DeepSeek Chat 模型实例。

    DeepSeek 提供与 OpenAI 兼容的接口，因此通过 langchain-openai 的
    ChatOpenAI 接入，只需将 base_url 指向 DeepSeek。

    Returns:
        ChatOpenAI: 配置好的 DeepSeek 聊天模型。

    Raises:
        ValueError: API Key 未配置时抛出。
    """
    # 先校验 Key，缺失时给出清晰中文提示
    api_key = config.ensure_api_key()

    return ChatOpenAI(
        model=config.DEEPSEEK_MODEL,
        api_key=api_key,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=0.3,  # 较低温度，回答更稳定、更贴合上下文
    )
