"""检索器构建"""

import os
import time as _time
from datetime import date, timedelta

import streamlit as st
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.vector_stores import (
    MetadataFilters, MetadataFilter, FilterOperator, FilterCondition,
)
from llama_index.core.schema import TextNode
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import cfg
from indexer import get_chroma_collection, extract_node_meta

try:
    import jieba

    def chinese_tokenizer(text: str):
        return list(jieba.cut_for_search(text))

    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

    def chinese_tokenizer(text: str):
        """没装 jieba 时降级：按空白切分。"""
        return text.split()

# ================= Rerank =================
@st.cache_resource(show_spinner=False)
def load_rerank():
    return SentenceTransformerRerank(
        model=cfg["retrieval"]["rerank_model"],
        top_n=cfg["retrieval"]["rerank_top_n"],
    )


# ================= BM25 节点 =================
@st.cache_resource(show_spinner=False)
def build_bm25_nodes(kb_name: str, count: int):
    collection = get_chroma_collection(kb_name)
    result = collection.get(include=["documents", "metadatas"])
    nodes = []
    if result["documents"]:
        for i, text in enumerate(result["documents"]):
            chroma_meta = result["metadatas"][i] if result["metadatas"] else {}
            chroma_meta = dict(chroma_meta) if chroma_meta else {}
            real_meta = extract_node_meta(chroma_meta)
            if not real_meta.get("file_name") and real_meta.get("file_path"):
                real_meta["file_name"] = os.path.basename(real_meta["file_path"])
            nodes.append(TextNode(text=text, metadata=real_meta))
    return nodes


# ================= 文件列表 =================
@st.cache_resource(show_spinner=False)
def get_available_files(kb_name: str, count: int):
    collection = get_chroma_collection(kb_name)
    result = collection.get(include=["metadatas"])
    files = set()
    for meta in (result.get("metadatas") or []):
        if not meta:
            continue
        real_meta = extract_node_meta(meta)
        fn = real_meta.get("file_name")
        if fn:
            files.add(fn)
    return sorted(files)


# ================= 节点数 =================
def get_node_count(kb_name: str):
    return get_chroma_collection(kb_name).count()


# ================= 混合检索器 =================
@st.cache_resource(show_spinner=False)
def build_hybrid_retriever(kb_name: str, count: int):
    nodes = build_bm25_nodes(kb_name, count)
    if not nodes:
        return None

    bm25_kwargs = {
        "nodes": nodes,
        "similarity_top_k": cfg["retrieval"]["similarity_top_k"],
        "tokenizer": chinese_tokenizer,
    }
    bm25_retriever = BM25Retriever.from_defaults(**bm25_kwargs)

    collection = get_chroma_collection(kb_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    tmp_index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
    vector_retriever = tmp_index.as_retriever(
        similarity_top_k=cfg["retrieval"]["similarity_top_k"],
    )

    return QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=cfg["retrieval"]["similarity_top_k"],
        num_queries=cfg["retrieval"].get("num_queries", 3),
        mode="reciprocal_rerank",
        use_async=cfg["retrieval"].get("use_async", True),
        verbose=False,
    )


# ================= 日期过滤 =================
def _since_ts(range_opt):
    if range_opt == "全部":
        return None
    days = {"今天": 1, "最近 3 天": 3, "最近 7 天": 7, "最近 30 天": 30}[range_opt]
    since = date.today() - timedelta(days=days - 1)
    return int(_time.mktime(since.timetuple()))


def _date_filter(range_opt):
    ts = _since_ts(range_opt)
    if ts is None:
        return None
    return MetadataFilter(key="date_ts", value=ts, operator=FilterOperator.GTE)


# ================= 构建检索器 =================
def build_retriever(index, hybrid_retriever, kb_name, selected_files, date_range="全部"):
    # 无过滤条件 → 直接用预构建的混合检索器（省一次 BM25 构建）
    if not selected_files and date_range == "全部":
        return hybrid_retriever if hybrid_retriever else index.as_retriever(
            similarity_top_k=cfg["retrieval"]["similarity_top_k"],
        )

    file_filters = [
        MetadataFilter(key="file_name", value=f, operator=FilterOperator.EQ)
        for f in selected_files
    ]
    date_f = _date_filter(date_range)

    if file_filters and date_f:
        filters = MetadataFilters(
            filters=[
                MetadataFilters(filters=file_filters, condition=FilterCondition.OR),
                date_f,
            ],
            condition=FilterCondition.AND,
        )
    elif file_filters:
        filters = MetadataFilters(filters=file_filters, condition=FilterCondition.OR)
    elif date_f:
        filters = MetadataFilters(filters=[date_f], condition=FilterCondition.AND)
    else:
        filters = None

    vector_retriever = index.as_retriever(
        similarity_top_k=cfg["retrieval"]["similarity_top_k"],
        filters=filters,
    )

    nodes = build_bm25_nodes(kb_name, get_chroma_collection(kb_name).count())
    selected_set = set(selected_files) if selected_files else None
    since_ts = _since_ts(date_range)

    filtered_nodes = []
    for n in nodes:
        if selected_set and n.metadata.get("file_name") not in selected_set:
            continue
        if since_ts is not None:
            n_ts = n.metadata.get("date_ts", 0) or 0
            if n_ts < since_ts:
                continue
        filtered_nodes.append(n)

    if filtered_nodes:
        bm25_kwargs = {
            "nodes": filtered_nodes,
            "similarity_top_k": cfg["retrieval"]["similarity_top_k"],
            "tokenizer": chinese_tokenizer,
        }
        bm25_retriever = BM25Retriever.from_defaults(**bm25_kwargs)
        return QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=cfg["retrieval"]["similarity_top_k"],
            num_queries=cfg["retrieval"].get("num_queries", 3),
            mode="reciprocal_rerank",
            use_async=cfg["retrieval"].get("use_async", True),
            verbose=False,
        )
    return vector_retriever