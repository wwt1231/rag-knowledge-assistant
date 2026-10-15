"""Chroma 连接、入库、索引加载"""

import os
import json
from pathlib import Path

import streamlit as st
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from config import (
    cfg, CHROMA_DIR, get_kb_config, logger,
)
from constants import SUPPORTED_EXTENSIONS, META_KEYS
from documents import load_any_document


def _build_llm():
    """根据 config.yaml 的 llm.provider 构建 LLM。"""
    provider = cfg["llm"].get("provider", "ollama")

    if provider == "deepseek":
        from llama_index.llms.deepseek import DeepSeek
        return DeepSeek(
            model=cfg["llm"]["model"],
            api_key=cfg["llm"].get("api_key") or os.environ.get("DEEPSEEK_API_KEY"),
            temperature=cfg["llm"]["temperature"],
            max_tokens=cfg["llm"].get("max_tokens", 2048),
            system_prompt=cfg["system_prompt"],
        )

    if provider == "ollama":
        from llama_index.llms.ollama import Ollama
        return Ollama(
            model=cfg["llm"]["model"],
            base_url=cfg["llm"]["base_url"],
            temperature=cfg["llm"]["temperature"],
            additional_kwargs={
                "num_ctx": cfg["llm"]["num_ctx"],
                "num_predict": cfg["llm"]["num_predict"],
            },
            system_prompt=cfg["system_prompt"],
        )

    raise ValueError(f"不支持的 LLM provider: {provider}")

# ================= 从 Chroma metadata 里解析真实 node 元数据 =================
def extract_node_meta(chroma_meta: dict) -> dict:
    """从 Chroma metadata 提取 node metadata（含新闻字段）。"""
    result = {}

    for key in META_KEYS:
        if chroma_meta.get(key) is not None:
            result[key] = chroma_meta[key]

    node_content = chroma_meta.get("_node_content")
    if node_content and isinstance(node_content, str):
        try:
            node_dict = json.loads(node_content)
            meta = node_dict.get("metadata") or {}
            for key in META_KEYS:
                if meta.get(key) is not None and key not in result:
                    result[key] = meta[key]
            for rel in (node_dict.get("relationships") or {}).values():
                rel_meta = rel.get("metadata") or {}
                for key in META_KEYS:
                    if rel_meta.get(key) is not None and key not in result:
                        result[key] = rel_meta[key]
                if result.get("file_name"):
                    break
        except Exception as e:
            logger.warning("解析 _node_content 失败: %s", e)

    return result


# ================= 已索引文件记录 =================
def rel_key(file_path: str, kb_name: str) -> str:
    """把绝对路径转成相对该知识库 data_dir 的路径。"""
    kb = get_kb_config(kb_name)
    try:
        return os.path.relpath(file_path, kb["data_dir"])
    except ValueError:
        return os.path.basename(file_path)


def load_indexed_files(kb_name: str) -> set:
    kb = get_kb_config(kb_name)
    path = kb["indexed_files_path"]
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data) if isinstance(data, list) else set()
    except (json.JSONDecodeError, OSError) as e:
        print(f"[警告] 读取 {path} 失败：{e}，按空记录处理。")
        return set()


def save_indexed_files(files: set, kb_name: str):
    kb = get_kb_config(kb_name)
    with open(kb["indexed_files_path"], "w", encoding="utf-8") as f:
        json.dump(list(files), f, ensure_ascii=False, indent=2)


# ================= Chroma Collection =================
@st.cache_resource(show_spinner=False)
def get_chroma_collection(kb_name: str):
    try:
        kb = get_kb_config(kb_name)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        return client.get_or_create_collection(kb["collection_name"])
    except Exception as e:
        st.error(f"连接 Chroma 失败：{e}")
        raise


# ================= 切分流水线 =================
@st.cache_resource(show_spinner=False)
def _get_splitter():
    return SentenceSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        include_metadata=False,
    )


@st.cache_resource(show_spinner=False)
def _get_markdown_parser():
    return MarkdownNodeParser()


@st.cache_resource(show_spinner=False)
def _get_pipeline(use_markdown_parser: bool, embed_model_name: str) -> IngestionPipeline:
     # embed_model_name 只用作缓存 key：换 embedding 模型时自动重建 pipeline
    parsers = []
    if use_markdown_parser:
        parsers.append(_get_markdown_parser())
    parsers.append(_get_splitter())
    parsers.append(Settings.embed_model)
    return IngestionPipeline(transformations=parsers)


def build_ingestion_pipeline(use_markdown_parser: bool) -> IngestionPipeline:
    return _get_pipeline(use_markdown_parser, cfg["embedding"]["model"])


# ================= 入库 =================
def ingest_document(file_path: str, index, indexed: set, kb_name: str) -> int:
    ext = os.path.splitext(file_path)[1].lower()
    logger.info("[%s] 开始解析 %s", kb_name, file_path)

    try:
        docs = load_any_document(file_path)
        logger.info("[%s] 解析得到 %d 页", kb_name, len(docs))

        pipeline = build_ingestion_pipeline(use_markdown_parser=(ext == ".pdf"))
        nodes = pipeline.run(documents=docs)
        logger.info("[%s] 切分得到 %d 个节点", kb_name, len(nodes))

        if not nodes:
            raise ValueError(f"未产生任何节点：{file_path}")

        BATCH = 50
        total = len(nodes)
        for i in range(0, total, BATCH):
            batch = nodes[i:i+BATCH]
            index.insert_nodes(batch)
            logger.info("[%s] 写入进度：%d/%d", kb_name, min(i+BATCH, total), total)

        indexed.add(rel_key(file_path, kb_name))
        save_indexed_files(indexed, kb_name)
        logger.info("[%s] 入库成功：%s，共 %d 个节点", kb_name, file_path, total)
        return total

    except Exception as e:
        logger.exception("[%s] 入库失败：%s", kb_name, file_path)
        raise


# ================= 遍历文件 =================
def iter_supported_files(kb_name: str):
    kb = get_kb_config(kb_name)
    for p in sorted(Path(kb["data_dir"]).rglob("*")):
        if not p.is_file():
            continue
        name = p.name
        if name.startswith(".") or name.startswith("~$"):
            continue
        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield p


# ================= 加载索引 =================
@st.cache_resource(show_spinner=False)
def load_index(kb_name: str):
    with st.spinner(f"正在加载「{get_kb_config(kb_name)['name']}」..."):
        Settings.llm = _build_llm()
        Settings.embed_model = OllamaEmbedding(
            model_name=cfg["embedding"]["model"],
            base_url=cfg["embedding"]["base_url"],
        )

        collection = get_chroma_collection(kb_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        files = list(iter_supported_files(kb_name))
        # 空知识库也构建一个空 index，让 UI 能正常渲染、侧边栏能切换
        # （不再 return None）

        index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )

        indexed = load_indexed_files(kb_name)

        if collection.count() > 0 and not indexed:
            logger.warning("[%s] indexed_files.json 丢失，尝试从 Chroma 重建记录", kb_name)
            rebuilt = set()
            result = collection.get(include=["metadatas"])
            for meta in (result.get("metadatas") or []):
                if not meta:
                    continue
                real_meta = extract_node_meta(meta)
                fp = real_meta.get("file_path")
                if fp:
                    rebuilt.add(rel_key(fp, kb_name))
            indexed = rebuilt
            save_indexed_files(indexed, kb_name)
            logger.info("[%s] 已从 Chroma 重建 %d 条文件记录", kb_name, len(indexed))

        to_index = [p for p in files if rel_key(str(p), kb_name) not in indexed]
        failed = []
        for f in to_index:
            try:
                ingest_document(str(f), index, indexed, kb_name)
            except Exception as e:
                failed.append((str(f), str(e)))
                logger.warning("[%s] 入库失败 %s: %s", kb_name, f, e)

        if failed:
            logger.warning("[%s] 共 %d 个文件入库失败", kb_name, len(failed))

        return index