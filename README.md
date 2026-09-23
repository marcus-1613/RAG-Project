# 📚 PDF 本地知识库问答系统（RAG）

> 一个基于 **RAG（Retrieval-Augmented Generation，检索增强生成）** 的 PDF 知识库问答系统。
> 上传 PDF → 自动切分、向量化入库 → 提问 → 检索最相关片段 → 结合 **DeepSeek** 大模型生成有据可依的回答。

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-✓-1C3C3C)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-FFDE59)
![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-4D6BFE)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)

> 本项目面向 **AI 大模型应用开发学习** 与 **实习求职**，代码遵循真实公司项目的工程化组织方式，同时保持清晰易懂。

---

## ✨ 核心特性

- 📄 上传单个或多个 **PDF** 文件，自动提取文本
- ✂️ 文本智能切分为文本块（chunk），保留上下文重叠
- 🧮 使用本地 **HuggingFace Embedding** 将文本块转为向量（无需额外 API Key）
- 💾 向量与原文持久化存储到 **ChromaDB**
- 🔍 提问后检索 **Top 5** 最相似文本块
- 🤖 拼接 RAG Prompt，调用 **DeepSeek** 生成回答
- 📑 展示最终答案 + 原文片段 + 来源文件名 + 页码
- ♻️ 基于文件哈希去重，重复 PDF 不会重复处理

---

## 🧭 系统架构

### 整体流程

```
        ┌───────────────────────── 建库（Ingestion）─────────────────────────┐
        │                                                                      │
  PDF ──▶ PyPDFLoader ──▶ RecursiveTextSplitter ──▶ HuggingFaceEmbedding ──┐  │
        │                                                                   │  │
        └───────────────────────────────────────────────────────────────────┼──┘
                                                                            ▼
                                                                ┌─────────────────────┐
                                                                │     ChromaDB        │
                                                                │   （向量 + 元数据）  │
                                                                └──────────┬──────────┘
        ┌───────────────────────── 问答（Query）───────────────────────────┐
        │                                                                    │
  问题 ──▶ 向量化 ──▶ 相似度检索 Top5 ──▶ 拼接 RAG Prompt ──▶ DeepSeek ──▶ 答案 │
        │                                                                    │
        └────────────────────────────────────────────────────────────────────┘
```

### 模块划分（`src/`）

| 模块 | 职责 |
|------|------|
| `config.py` | 集中配置（路径、模型、参数），从 `.env` 读取 |
| `loader.py` | 使用 PyPDFLoader 加载 PDF，提取文本与元数据 |
| `splitter.py` | RecursiveCharacterTextSplitter 文本切分 |
| `embeddings.py` | HuggingFace 本地 Embedding 模型（缓存复用） |
| `vector_store.py` | ChromaDB 持久化、增删、查询 |
| `retriever.py` | Top-K 相似度检索 |
| `llm.py` | DeepSeek 模型接入（ChatOpenAI + 自定义 base_url） |
| `dedup.py` | 文件哈希去重（文件名 + SHA-256） |
| `rag_chain.py` | RAG 业务编排（检索 → Prompt → 生成） |

---

## 🛠️ 技术栈

| 组件 | 说明 |
|------|------|
| Python | 3.11+ |
| LangChain | 文档加载、切分、检索编排 |
| PyPDFLoader | PDF 文本提取 |
| RecursiveCharacterTextSplitter | 文本切分 |
| HuggingFace Embedding | 本地中文语义向量（`BAAI/bge-small-zh-v1.5`） |
| ChromaDB | 向量存储与检索（持久化） |
| DeepSeek API | 大语言模型（`deepseek-chat`） |
| Streamlit | Web 交互界面 |

---

## 📁 目录结构

```
RAG-Project/
├── app.py                      # Streamlit 页面（上传、问答、展示）
├── requirements.txt            # 依赖清单
├── README.md                   # 本文档
├── .env.example                # 环境变量模板
├── .gitignore                  # Git 忽略规则
│
├── data/
│   ├── pdfs/                   # 存放 PDF 文件（默认已忽略）
│   │   └── README.md
│   └── chroma/                 # ChromaDB 持久化目录（运行时生成）
│
├── src/
│   ├── __init__.py
│   ├── config.py               # 集中配置（参数、路径、模型）
│   ├── loader.py               # PDF 加载
│   ├── splitter.py             # 文本切分
│   ├── embeddings.py           # Embedding 模型
│   ├── vector_store.py         # ChromaDB 持久化与增删
│   ├── retriever.py            # Top-K 检索
│   ├── llm.py                  # DeepSeek 模型
│   ├── dedup.py                # PDF 去重（文件哈希）
│   └── rag_chain.py            # RAG 业务编排
│
├── scripts/
│   ├── generate_test_pdf.py    # 生成测试 PDF
│   └── verify_rag.py           # 命令行验证脚本
│
└── tests/
    ├── conftest.py             # 测试夹具
    └── test_rag.py             # 核心流程测试
```

---

## 🚀 快速开始

### 1. 创建虚拟环境

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> 首次安装会下载 `torch`、`sentence-transformers` 等较大包，耗时较长，请耐心等待。

### 3. 配置环境变量

```bash
# 复制模板（Windows 用 copy .env.example .env）
cp .env.example .env
```

编辑 `.env`：

```dotenv
DEEPSEEK_API_KEY=你的_API_Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

> API Key 可在 https://platform.deepseek.com 申请。

### 4. 启动

```bash
streamlit run app.py
```

浏览器会自动打开页面。

---

## 🖥️ 使用方式

1. 在左侧上传 PDF，点击「建立 / 更新知识库」
2. 等待入库完成（有进度提示）
3. 主区域输入问题，点击「开始问答」
4. 查看最终答案、检索到的原文片段与来源

> ⚠️ **注意**：目前仅支持**文字版** PDF（可选中复制文字的 PDF）。纯扫描图片版需先通过 OCR 转成文字版。

---

## 🧪 命令行验证（可选）

不启动界面也能验证全链路：

```bash
# 生成测试 PDF
python scripts/generate_test_pdf.py

# 建库 + 检索（无需 API Key）
python scripts/verify_rag.py --build --query "什么是RAG"

# 建库 + 检索 + 调用 DeepSeek 回答（需 API Key）
python scripts/verify_rag.py --build --query "什么是RAG" --answer
```

---

## ⚙️ 配置说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | 1000 | 每个文本块的最大字符数 |
| `chunk_overlap` | 200 | 相邻文本块之间的重叠字符数 |

- **chunk_size 太小** → 上下文不完整，知识点易被拆散；**太大** → 检索粒度变粗，可能混入无关内容。
- **chunk_overlap** 用于保留块间上下文，避免关键信息恰好被切在边界处而丢失。

统一在 `src/config.py` 中配置，也可通过 `.env` 的 `CHUNK_SIZE`、`CHUNK_OVERLAP` 覆盖。

---

## ❓ FAQ

**Q1：为什么用本地 Embedding，而不是 DeepSeek？**
DeepSeek 目前主要提供对话（chat）类模型，没有可直接使用的 embedding 接口。使用本地 HuggingFace 中文模型（`bge-small-zh-v1.5`）无需额外 API Key，且中文语义检索效果良好。

**Q2：首次运行 Embedding 很慢或报网络错误？**
首次需从 HuggingFace 下载模型（约 100MB）。国内网络较慢时，可在 `.env` 中设置镜像：

```dotenv
HF_ENDPOINT=https://hf-mirror.com
```

**Q3：上传的 PDF 没有提取出文字？**
很可能是纯扫描图片版 PDF，本项目暂不支持 OCR，请使用文字版 PDF。

**Q4：重复上传同一个 PDF 会怎样？**
系统通过「文件名 + 内容哈希」判断是否已入库，重复文件会被跳过，不会重复生成 Embedding。

---

## 🚧 当前限制

1. 仅支持文字版 PDF，不支持扫描版（需 OCR）
2. 本地 CPU 运行 Embedding，大批量入库速度有限
3. 基于简单相似度检索，未实现重排序（rerank）
4. 暂不支持多轮对话（每次问答独立）
5. 暂未处理 PDF 中的表格、图片等复杂排版

## 🗺️ 路线图

- [ ] 接入 OCR，支持扫描版 PDF
- [ ] 引入 rerank 模型，提升检索精度
- [ ] 支持多轮对话，结合历史上下文
- [ ] 使用更强的 Embedding 模型（如 `bge-large-zh`）
- [ ] 增加知识库管理（删除、更新单个文档）
- [ ] 支持更多文件格式（Word、Markdown、网页等）
- [ ] 使用 GPU 加速 Embedding

---

## 📄 许可证

本项目暂未声明许可证。如需开源，建议添加一份 `LICENSE`（如 MIT）。
