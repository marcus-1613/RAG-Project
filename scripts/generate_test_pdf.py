"""生成用于测试的 PDF 文件（含中文）。

生成一个包含几段中文内容的文字版 PDF，方便验证加载、切分、检索全流程。
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


# 测试文档内容（分页，每页一段主题）
SECTIONS = [
    (
        "人工智能简介",
        "人工智能（Artificial Intelligence，简称 AI）是研究、开发用于模拟、"
        "延伸和扩展人的智能的理论、方法、技术及应用系统的一门技术科学。"
        "人工智能的目标是让机器能够完成通常需要人类智能才能完成的任务，"
        "例如视觉感知、语音识别、决策和语言翻译等。",
    ),
    (
        "机器学习概述",
        "机器学习（Machine Learning）是人工智能的一个核心分支，"
        "它让计算机能够从数据中自动学习规律，而无需进行显式的编程。"
        "机器学习主要包括监督学习、无监督学习和强化学习三大类。"
        "监督学习使用带标签的数据进行训练，无监督学习则在无标签数据中寻找结构。",
    ),
    (
        "深度学习与神经网络",
        "深度学习（Deep Learning）是机器学习的一个重要方向，"
        "它基于多层神经网络来学习数据的复杂表示。"
        "深度神经网络由多个隐藏层组成，能够自动提取从低级到高级的特征。"
        "卷积神经网络（CNN）擅长处理图像，循环神经网络（RNN）适合处理序列数据，"
        "而 Transformer 架构则在自然语言处理领域取得了巨大成功。",
    ),
    (
        "RAG 检索增强生成",
        "RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合了"
        "信息检索与文本生成的技术。它先从外部知识库中检索与问题相关的文档片段，"
        "然后将这些片段作为上下文提供给大语言模型，让模型基于真实资料生成答案。"
        "RAG 可以有效缓解大模型的知识幻觉问题，让回答更具事实依据和可追溯性。",
    ),
    (
        "向量数据库与 Embedding",
        "向量数据库用于存储和检索高维向量数据。Embedding 是将文本等非结构化数据"
        "转换为稠密向量的过程，语义相近的文本在向量空间中距离也更近。"
        "ChromaDB 是一个轻量级的开源向量数据库，常用于 RAG 系统中存储文本块的向量。"
        "检索时通过计算问题向量与库中向量的相似度，找出最相关的文本块。",
    ),
]


def _register_chinese_font() -> str:
    """注册系统中文字体，返回字体名。

    优先使用微软雅黑（msyh.ttc），找不到则回退到黑体（simhei.ttf）。
    """
    candidates = [
        ("C:/Windows/Fonts/msyh.ttc", "MSYH"),
        ("C:/Windows/Fonts/simhei.ttf", "SIMHEI"),
    ]
    for font_path, font_name in candidates:
        if Path(font_path).exists():
            try:
                pdfmetrics.registerFont(TTFont(font_name, font_path))
                return font_name
            except Exception:  # noqa: BLE001 - 尝试下一个字体
                continue
    # 都不存在时用内置 Helvetica（中文会乱码，但流程仍可跑通）
    return "Helvetica"


def generate_test_pdf(output_path: str | Path) -> Path:
    """生成测试 PDF 文件。

    Args:
        output_path: 输出 PDF 路径。

    Returns:
        Path: 生成的 PDF 文件路径。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    font_name = _register_chinese_font()

    c = canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4

    for title, content in SECTIONS:
        # 标题
        c.setFont(font_name, 18)
        c.drawString(72, height - 72, title)

        # 正文（简单按固定宽度换行）
        c.setFont(font_name, 12)
        text_obj = c.beginText(72, height - 120)
        text_obj.setFont(font_name, 12)
        text_obj.setLeading(22)

        chars_per_line = 26
        for i in range(0, len(content), chars_per_line):
            text_obj.textLine(content[i : i + chars_per_line])

        c.drawText(text_obj)
        c.showPage()

    c.save()
    return output_path


if __name__ == "__main__":
    out = generate_test_pdf("data/pdfs/AI知识库测试文档.pdf")
    print(f"测试 PDF 已生成：{out}")
