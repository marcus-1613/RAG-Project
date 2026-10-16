"""Rerank 模块（Cross-Encoder 精排，可选）。

它在流程里的位置：**召回之后、生成之前。**

    向量检索 / BM25 / RRF 融合解决的都是"**召回**"问题——让正确的片段
    进入候选集。但候选集内部的**顺序**不一定对：RRF 只用到了排名信息，
    并没有真正逐条比较"这个问题"和"这段文本"有多相关。

Rerank 用 Cross-Encoder 做这件事：把 (问题, 候选片段) **成对**送进模型打分。

    对比一下两种模型的差别：
    - 双塔（Bi-Encoder，即 Embedding）：问题和文档**分别**编码成向量，
      再算余弦。文档向量可以预先算好，所以快，适合海量召回。
      代价是编码时看不到对方，语义交互丢失。
    - 交叉编码（Cross-Encoder）：问题和文档**拼在一起**送进模型，
      注意力机制能直接建模两者之间的交互，精度明显更高。
      代价是每对都要跑一次前向，无法预计算，慢。

    所以正确用法是：先用双塔从全库召回几十条（快），再用 Cross-Encoder
    精排出前几条（准）。本模块只对少量候选做精排，正是这个分工。

降级策略：
    模型加载失败（未安装依赖、离线无法下载、内存不足）时**不阻断流程**，
    直接返回原顺序并给出提示。检索质量退回到融合排序的效果，但问答仍可用。
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# 中文重排模型（与 Embedding 同属 BGE 系列，中文场景表现稳定）
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"


class Reranker:
    """Cross-Encoder 精排器。

    模型采用懒加载：只有真正调用 rerank 时才尝试载入，
    避免在不需要精排的场景下白白占用内存。
    """

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL) -> None:
        self.model_name = model_name
        self._model: Any = None
        self._load_failed = False
        self._fail_reason = ""

    @property
    def available(self) -> bool:
        """当前是否可用（尚未尝试加载时按可用处理）。"""
        return not self._load_failed

    @property
    def fail_reason(self) -> str:
        """加载失败的原因，供界面提示使用。"""
        return self._fail_reason

    def _ensure_model(self) -> Any:
        """懒加载 Cross-Encoder 模型；失败时记录原因并返回 None。"""
        if self._model is not None or self._load_failed:
            return self._model

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, max_length=512)
        except Exception as exc:  # noqa: BLE001 - 任何失败都降级，不阻断问答
            self._load_failed = True
            self._fail_reason = str(exc)
            logger.warning("Rerank 模型加载失败，已降级为融合排序：%s", exc)
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int,
    ) -> list[tuple[Document, float]]:
        """对候选文本块按与问题的相关性重新排序。

        Args:
            query: 用户问题。
            documents: 候选文本块（通常来自混合检索，数量不多）。
            top_n: 精排后保留的数量。

        Returns:
            list[tuple[Document, float]]: (文本块, 相关性分) 列表，降序。
                模型不可用时返回原顺序，分数以 0.0 占位，保证调用方
                无需区分两种情况。
        """
        if not documents:
            return []

        model = self._ensure_model()
        if model is None:
            return [(doc, 0.0) for doc in documents[:top_n]]

        pairs = [(query, doc.page_content) for doc in documents]
        try:
            scores = model.predict(pairs)
        except Exception as exc:  # noqa: BLE001 - 预测失败同样降级
            logger.warning("Rerank 预测失败，已降级为融合排序：%s", exc)
            return [(doc, 0.0) for doc in documents[:top_n]]

        ranked = sorted(
            zip(documents, (float(s) for s in scores)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:top_n]


# 进程内共享一个实例，避免重复加载模型
_shared_reranker: Reranker | None = None


def get_reranker(model_name: str | None = None) -> Reranker:
    """获取进程内共享的 Reranker 实例。

    Args:
        model_name: 模型名；为 None 时使用默认模型。

    Returns:
        Reranker: 共享实例。
    """
    global _shared_reranker
    if _shared_reranker is None or (
        model_name is not None and _shared_reranker.model_name != model_name
    ):
        _shared_reranker = Reranker(model_name or DEFAULT_RERANK_MODEL)
    return _shared_reranker
