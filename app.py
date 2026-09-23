"""Streamlit 页面入口。

只负责：文件上传、按钮、页面状态、展示答案和检索结果。
RAG 业务逻辑全部在 src/ 下的模块中实现。
"""

from __future__ import annotations

import shutil
from pathlib import Path

import streamlit as st

from src import config
from src.dedup import compute_file_hash
from src.loader import load_pdf
from src.splitter import split_documents
from src.vector_store import add_documents, get_existing_sources
from src.rag_chain import ask

# 确保数据目录存在
config.PDF_DIR.mkdir(parents=True, exist_ok=True)
config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="PDF 本地知识库问答系统",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PDF 本地知识库问答系统")
st.caption("基于 RAG（Retrieval-Augmented Generation）的本地知识库问答")


# ============================================================
# 侧边栏：PDF 上传与知识库管理
# ============================================================

with st.sidebar:
    st.header("1. 上传 PDF")
    uploaded_files = st.file_uploader(
        "选择 PDF 文件（可多选）",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("📥 建立 / 更新知识库", use_container_width=True):
        if not uploaded_files:
            st.warning("请先上传至少一个 PDF 文件。")
        else:
            existing_sources = get_existing_sources()

            for uploaded in uploaded_files:
                file_name = uploaded.name
                save_path = config.PDF_DIR / file_name

                # 1. 保存文件
                with st.status(f"正在保存 {file_name} ..."):
                    with save_path.open("wb") as f:
                        f.write(uploaded.getbuffer())

                # 2. 判断是否已入库（文件名 + 内容 hash）
                file_hash = compute_file_hash(save_path)
                if file_name in existing_sources:
                    st.info(f"⏭️ {file_name} 已入库，跳过（不重复处理）。")
                    continue

                try:
                    # 3. 读取 PDF
                    with st.status(f"正在读取 PDF：{file_name} ...", expanded=True) as status:
                        status.write("正在读取 PDF...")
                        documents = load_pdf(save_path)
                        status.write(f"读取到 {len(documents)} 页。")

                        # 4. 切分
                        status.write("正在切分文本...")
                        chunks = split_documents(documents)
                        status.write(f"切分为 {len(chunks)} 个文本块。")

                        # 5. 写入 ChromaDB（内部会做 Embedding）
                        status.write("正在生成 Embedding 并写入 ChromaDB...")
                        add_documents(chunks)
                        status.update(
                            label=f"✅ {file_name} 入库完成（{len(chunks)} 个文本块）",
                            state="complete",
                        )

                except ValueError as exc:
                    st.error(f"处理失败：{file_name}\n{exc}")
                except Exception as exc:  # noqa: BLE001
                    st.error(
                        f"处理 {file_name} 时发生未知错误：{exc}\n"
                        "请检查 PDF 是否为文字版，或查看控制台日志。"
                    )

    st.divider()
    st.header("2. 知识库状态")

    existing = get_existing_sources()
    if existing:
        st.success(f"已入库 PDF 数量：{len(existing)}")
        for name in sorted(existing):
            st.markdown(f"- {name}")
    else:
        st.info("知识库为空，请先上传 PDF。")


# ============================================================
# 主区域：问答
# ============================================================

st.header("💬 提问")

question = st.text_input("输入你的问题：", placeholder="例如：这份文档主要讲了什么？")

if st.button("🔍 开始问答", type="primary"):
    if not question.strip():
        st.warning("请输入问题。")
    elif not get_existing_sources():
        st.warning("知识库为空，请先在左侧上传 PDF 并建立知识库。")
    else:
        try:
            with st.status("正在检索并生成回答...", expanded=True) as status:
                status.write("正在检索...")
                status.write("正在调用 DeepSeek...")
                result = ask(question)
                status.update(label="✅ 回答完成", state="complete")

            # 展示最终答案
            st.subheader("📝 最终答案")
            st.markdown(result.answer)

            # 展示检索到的原文片段和来源
            st.divider()
            st.subheader(f"📄 检索到的原文片段（Top {len(result.retrieved_documents)}）")

            for i, doc in enumerate(result.retrieved_documents, 1):
                source = doc.metadata.get("source", "未知来源")
                page = doc.metadata.get("page", "未知页")
                with st.expander(
                    f"片段 {i} — 来源：{source} ｜ 页码：{page}"
                ):
                    st.markdown(doc.page_content)

        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(
                f"问答失败：{exc}\n"
                "可能是 DeepSeek API Key 未配置或网络异常，请检查 .env 配置。"
            )
