\# 知识助手（RAG Knowledge Assistant）



基于 LlamaIndex + ChromaDB + Streamlit 的多知识库 RAG 问答系统，

支持本地 Ollama 与云端 DeepSeek 双后端切换。



\## 功能特性



\- 多知识库管理

\- 混合检索：向量 + BM25 + Rerank

\- 双 LLM 后端：本地 Ollama / 云端 DeepSeek

\- 支持 PDF、PPTX、DOCX、CSV、Excel、Markdown、HTML

\- 引用溯源，侧边栏折叠查看原文

\- 文件级、时间范围级检索过滤



\## 技术栈



| 组件 | 说明 |

|:---|:---|

| 框架 | LlamaIndex |

| 向量库 | ChromaDB |

| 界面 | Streamlit |

| Embedding | bge-m3（本地 Ollama） |

| Rerank | bge-reranker-base |

| LLM（本地） | qwen2.5:7b |

| LLM（云端） | DeepSeek |



\## 快速开始



\\`\\`\\`bash

pip install -r requirements.txt

ollama pull qwen2.5:7b

ollama pull bge-m3

streamlit run app.py

\\`\\`\\`



\## License



MIT

