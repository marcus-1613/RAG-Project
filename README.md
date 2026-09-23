# PDF 本地知识库问答系统（RAG）

一个基于 **RAG（Retrieval-Augmented Generation，检索增强生成）** 的 PDF 知识库问答系统。

用户可以上传 PDF 文件，系统自动完成文本提取、切分、向量化与入库；之后用户提问，系统从知识库检索最相关的文本片段，结合 **DeepSeek** 大模型生成有据可依的回答，并展示原文片段和来源信息。

> 本项目面向 **AI 大模型应用开发学习** 与 **实习求职**，代码遵循真实公司项目的工程化组织方式，同时保持清晰易懂。

---

## 一、项目功能

- 📄 上传单个或多个 PDF 文件
- ✂️ 自动读取 PDF 文本并切分为文本块（chunk）
- 🧮 使用本地 Embedding 模型将文本块转换为向量
- 💾 向量与原文持久化保存到 ChromaDB
- 🔍 用户提问后检索 Top 5 最相似文本块
- 🤖 拼接 RAG Prompt，调用 DeepSeek 生成回答
- 📑 展示最终答案 + 检索到的原文片段 + 来源文件名 + 页码
- ♻️ 通过文件哈希去重，重复上传的 PDF 不会重复处理

---

## 二、技术栈

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

## 三、项目结构

```
rag-pdf-qa/
├── app.py                      # Streamlit 页面（上传、问答、展示）
├── requirements.txt            # 依赖清单
├── README.md                   # 本文档
├── .env.example                # 环境变量模板
├── .gitignore                  # Git 忽略规则
│
├── data/
│   ├── pdfs/                   # 存放 PDF 文件
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

## 四、RAG 工作流程

```
PDF
 ↓  PyPDFLoader（loader.py）
Document（含文件名、页码 metadata）
 ↓  RecursiveCharacterTextSplitter（splitter.py）
Chunks（文本块）
 ↓  HuggingFace Embedding（embeddings.py）
向量
 ↓  写入（vector_store.py）
ChromaDB（持久化）
 ↓  检索（retriever.py）
Top 5 文本块
 ↓  构造 Prompt（rag_chain.py）
RAG Prompt
 ↓  调用（llm.py）
DeepSeek
 ↓
Answer（最终回答）
```

---

## 五、环境安装

### 1. 创建虚拟环境

```bash
# Windows（本项目使用）
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

> 首次安装会下载 `torch`、`sentence-transformers` 等较大的包，耗时较长，请耐心等待。

### 3. 配置 .env

复制模板并填入你的 API Key：

```bash
cp .env.example .env   # Windows 用 copy .env.example .env
```

编辑 `.env`：

```
DEEPSEEK_API_KEY=你的_API_Key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

> API Key 可在 https://platform.deepseek.com 申请。

---

## 六、准备 PDF

将 **文字版** PDF 文件放入 `data/pdfs/` 目录（也支持通过网页直接上传）。

> ⚠️ **注意**：目前仅支持文字版 PDF（可选中复制文字的 PDF）。
> 纯扫描图片版 PDF（拍照/扫描生成的图片型 PDF）暂不支持，需先通过 OCR 转成文字版。

---

## 七、启动 Streamlit

```bash
streamlit run app.py
```

启动后浏览器会自动打开页面，流程如下：

1. 在左侧上传 PDF，点击「建立 / 更新知识库」
2. 等待入库完成（会有进度提示）
3. 在主区域输入问题，点击「开始问答」
4. 查看最终答案和检索到的原文片段

---

## 八、命令行验证（可选）

不启动界面也能验证全链路：

```bash
# 生成一个测试 PDF
python scripts/generate_test_pdf.py

# 建库 + 检索（不需要 API Key）
python scripts/verify_rag.py --build --query "什么是RAG"

# 建库 + 检索 + 调用 DeepSeek 回答（需要 API Key）
python scripts/verify_rag.py --build --query "什么是RAG" --answer
```

---

## 九、chunk_size 与 chunk_overlap 说明

- **chunk_size（默认 1000）**：每个文本块的最大字符数。
  - 太小 → 上下文不完整，一个知识点可能被拆散。
  - 太大 → 检索粒度变粗，可能混入无关内容。
- **chunk_overlap（默认 200）**：相邻文本块之间的重叠字符数。
  - 用于保留相邻块之间的上下文，避免关键信息恰好被切在边界处而丢失。

这两个参数统一在 `src/config.py` 中配置（也可通过 `.env` 的 `CHUNK_SIZE`、`CHUNK_OVERLAP` 覆盖）。

---

## 十、RAG 常见问题

**Q1：为什么用本地 Embedding，而不是 DeepSeek？**
DeepSeek 目前主要提供对话（chat）类模型，没有可直接使用的 embedding 接口。使用本地 HuggingFace 中文模型（`bge-small-zh-v1.5`）无需额外 API Key，且对中文语义检索效果良好。

**Q2：首次运行 Embedding 很慢或报网络错误？**
首次需要从 HuggingFace 下载模型（约 100MB）。国内网络较慢时，可在 `.env` 中设置镜像：

```
HF_ENDPOINT=https://hf-mirror.com
```

**Q3：上传的 PDF 没有被提取出文字？**
很可能是纯扫描图片版 PDF。本项目暂不支持 OCR，请使用文字版 PDF。

**Q4：重复上传同一个 PDF 会怎样？**
系统通过文件名 + 文件内容哈希判断是否已入库，重复文件会被跳过，不会重复生成 Embedding。

---

## 十一、项目目前的限制

1. 仅支持文字版 PDF，不支持扫描版（需 OCR）。
2. 使用本地 CPU 运行 Embedding，大批量 PDF 入库速度有限。
3. 向量库基于简单相似度检索，未实现重排序（rerank）。
4. 暂不支持多轮对话（每次问答独立）。
5. 暂未处理 PDF 中的表格、图片等复杂排版。

---

## 十二、后续优化方向

1. 接入 OCR，支持扫描版 PDF。
2. 引入 rerank 模型，提升检索精度。
3. 支持多轮对话，结合历史上下文。
4. 使用更强的 Embedding 模型（如 `bge-large-zh`）。
5. 增加知识库管理（删除、更新单个文档）。
6. 支持更多文件格式（Word、Markdown、网页等）。
7. 使用 GPU 加速 Embedding。
