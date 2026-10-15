"""主入口：UI + 聊天循环"""

import os
import re
import json
import shutil

import streamlit as st

# config 模块不含 st 调用，先导入拿 cfg，用于 set_page_config
from config import cfg

st.set_page_config(
    page_title=cfg.get("app_title", "知识助手"),
    page_icon=cfg.get("app_icon", "📚"),
    layout="centered",
)

# 其它 import 放在 set_page_config 之后
import chromadb

from config import (
    CHROMA_DIR, IMAGE_DIR,
    get_kb_config, ensure_dirs,
)
from constants import SUPPORTED_EXTENSIONS
from indexer import (
    load_index, load_indexed_files,
    get_chroma_collection, ingest_document, rel_key,
    extract_node_meta,
)
from retriever import (
    load_rerank, build_bm25_nodes, get_available_files, get_node_count,
    build_hybrid_retriever, build_retriever, _HAS_JIEBA,
)
from chat import build_chat_engine



# ================= 从 Chroma 反查补齐 source_node 的 metadata =================
def backfill_source_metadata(source_nodes, kb_name):
    """向量分支的 node metadata 是空的，从 Chroma 批量反查补齐。"""
    if not source_nodes:
        return source_nodes

    collection = get_chroma_collection(kb_name)
    node_ids = [n.node_id for n in source_nodes]

    try:
        r = collection.get(ids=node_ids, include=["metadatas"])
    except Exception:
        return source_nodes

    id_to_meta = {}
    for i, nid in enumerate(r.get("ids") or []):
        raw = r["metadatas"][i] if r.get("metadatas") else {}
        real = extract_node_meta(dict(raw) if raw else {})
        if real:
            id_to_meta[nid] = real

    for node in source_nodes:
        meta = id_to_meta.get(node.node_id)
        if meta:
            node.metadata.update(meta)

    return source_nodes

# ================= 清理 LLM 末尾的纯引用编号行 =================
def strip_trailing_citations(text: str) -> str:
    """删掉回答末尾单独成行的纯引用编号，如 [1][2] 或 [1] [2]。
    只删末尾连续几行；正文里的 [1] 不动。
    """
    lines = text.rstrip().split("\n")
    while lines:
        last = lines[-1].strip()
        # 匹配纯编号行："[1]" "[1][2]" "[1] [2]" "[1][2][3]"
        if re.fullmatch(r"(\[\d+\]\s*)+", last):
            lines.pop()
        else:
            break
    return "\n".join(lines)


# ================= 页面标题 =================
st.title(cfg.get("app_title", "知识助手"))
ensure_dirs()


# ================= 会话状态初始化 =================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": cfg.get("welcome_message", "你好，请问你想了解什么？"),
        }
    ]

if "index_version" not in st.session_state:
    st.session_state.index_version = 0

if "current_kb" not in st.session_state:
    kbs = cfg.get("knowledge_bases", {})
    st.session_state.current_kb = list(kbs.keys())[0] if kbs else "default"


# ================= 全局加载 =================
_kb_name = st.session_state.current_kb
_kb = get_kb_config(_kb_name)
index = load_index(_kb_name)
# 注意：空知识库不再 st.stop()，否则侧边栏会被挡住，用户无法切换


rerank = load_rerank()


# ================= 构建混合检索器 =================
hybrid_retriever = None
try:
    hybrid_retriever = build_hybrid_retriever(
        _kb_name, get_chroma_collection(_kb_name).count()
    )
except Exception as e:
    st.warning(f"混合检索构建失败，将回退到纯向量检索。\n\n错误：{e}")
    hybrid_retriever = None


# ================= 侧边栏：知识库切换 =================
with st.sidebar:
    st.subheader("🗂️ 知识库切换")
    kbs = cfg.get("knowledge_bases", {})
    kb_options = {k: v.get("name", k) for k, v in kbs.items()}

    if kb_options:
        kb_keys = list(kb_options.keys())
        current_idx = kb_keys.index(st.session_state.current_kb) \
            if st.session_state.current_kb in kb_keys else 0

        selected_kb = st.selectbox(
            "选择知识库",
            options=kb_keys,
            format_func=lambda k: kb_options[k],
            index=current_idx,
        )

        if selected_kb != st.session_state.current_kb:
            st.session_state.current_kb = selected_kb
            st.session_state.index_version += 1
            st.session_state.pop("chat_engine", None)
            st.session_state.pop("chat_state", None)
            st.session_state.pop("selected_files", None)
            st.rerun()

    st.divider()


# ================= 侧边栏：知识库管理 =================
with st.sidebar:
    st.subheader("📚 知识库管理")

    node_count = get_node_count(_kb_name)
    st.caption(f"当前节点数：{node_count}")

    if _HAS_JIEBA:
        st.caption("✅ BM25 已启用 jieba 中文分词")
    else:
        st.caption("⚠️ 未检测到 jieba，BM25 中文检索效果会打折扣")

    upload_types = [e.lstrip(".") for e in SUPPORTED_EXTENSIONS]
    uploaded_files = st.file_uploader(
        "上传新文档",
        type=upload_types,
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("解析并加入知识库"):
        kb_name_now = st.session_state.current_kb
        kb_now = get_kb_config(kb_name_now)
        indexed = load_indexed_files(kb_name_now)
        success_count = 0

        for uploaded_file in uploaded_files:
            safe_name = os.path.basename(uploaded_file.name)
            ext = os.path.splitext(safe_name)[1].lower()

            if ext not in SUPPORTED_EXTENSIONS:
                st.error(f"❌ 不支持的文件类型：{ext}")
                continue

            file_path = os.path.join(kb_now["data_dir"], safe_name)
            rk = rel_key(file_path, kb_name_now)

            if rk in indexed or os.path.exists(file_path):
                st.warning(f"⚠️ {safe_name} 已存在，跳过。")
                continue

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"正在解析 {safe_name}..."):
                try:
                    n = ingest_document(file_path, index, indexed, kb_name_now)
                    success_count += 1
                    st.success(f"✅ {safe_name} 已加入（新增 {n} 个节点）")
                except Exception as e:
                    st.error(f"❌ {safe_name} 入库失败：{e}")

        if success_count > 0:
            st.session_state.index_version += 1
            st.rerun()

    st.divider()

    st.checkbox(
        "在检索结果中显示图片（可能很长）",
        value=False,
        key="sidebar_show_images",
    )

    if st.button("🗑️ 清空对话"):
        st.session_state.messages = [
            {"role": "assistant", "content": cfg.get("welcome_message", "你好")}
        ]
        st.session_state.pop("chat_engine", None)
        st.session_state.pop("chat_state", None)
        st.rerun()

    if "confirm_rebuild" not in st.session_state:
        st.session_state.confirm_rebuild = False

    if st.button("🔍 扫描当前库的新文件"):
        st.cache_resource.clear()
        st.session_state.index_version += 1
        st.rerun()

    if st.button("🔍 扫描所有库的新文件"):
        st.cache_resource.clear()
        _kbs = cfg.get("knowledge_bases", {})
        _progress = st.progress(0)
        _status = st.empty()
        _failed = []
        _scanned = []

        for _i, _kb_key in enumerate(_kbs.keys()):
            _name = _kbs[_kb_key].get("name", _kb_key)
            _status.text(f"正在扫描「{_name}」...")
            try:
                _result = load_index(_kb_key)
                _scanned.append(_kb_key)
            except Exception as e:
                _failed.append((_kb_key, str(e)))
            _progress.progress((_i + 1) / len(_kbs))

        _progress.empty()
        _status.empty()

        if _failed:
            for _k, _e in _failed:
                st.warning(f"「{_k}」扫描失败：{_e}")
        else:
            st.success(f"✅ 已扫描 {len(_scanned)} 个知识库")

        st.session_state.index_version += 1
        st.rerun()

    if not st.session_state.confirm_rebuild:
        if st.button("🔄 重建索引"):
            st.session_state.confirm_rebuild = True
            st.rerun()
    else:
        st.warning(f"⚠️ 确认重建「{_kb['name']}」？")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("确认", type="primary"):
                kb_name_now = st.session_state.current_kb
                kb_now = get_kb_config(kb_name_now)
                st.cache_resource.clear()

                try:
                    client = chromadb.PersistentClient(path=CHROMA_DIR)
                    try:
                        client.delete_collection(kb_now["collection_name"])
                    except Exception:
                        pass
                    client.get_or_create_collection(kb_now["collection_name"])
                except Exception as e:
                    st.error(f"❌ 删除向量库失败：{e}")

                shutil.rmtree(IMAGE_DIR, ignore_errors=True)
                os.makedirs(IMAGE_DIR, exist_ok=True)
                if os.path.exists(kb_now["indexed_files_path"]):
                    os.remove(kb_now["indexed_files_path"])

                st.session_state.index_version += 1
                st.session_state.confirm_rebuild = False
                st.rerun()
        with col2:
            if st.button("取消"):
                st.session_state.confirm_rebuild = False
                st.rerun()


# ================= 侧边栏：时间范围、限定检索范围 =================
with st.sidebar:
    st.divider()
    st.subheader("🗓️ 时间范围")
    st.radio(
        "只看最近",
        ["全部", "今天", "最近 3 天", "最近 7 天", "最近 30 天"],
        index=0,
        horizontal=True,
        key="date_range",
    )

    st.subheader("📂 限定检索范围")
    all_files = get_available_files(
        _kb_name, get_chroma_collection(_kb_name).count()
    )

    if "selected_files" in st.session_state:
        valid = set(all_files)
        current = st.session_state.selected_files or []
        cleaned = [f for f in current if f in valid]
        if cleaned != current:
            st.session_state.selected_files = cleaned

    if all_files:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("全选", use_container_width=True):
                st.session_state.selected_files = all_files
                st.rerun()
        with col2:
            if st.button("清空", use_container_width=True):
                st.session_state.selected_files = []
                st.rerun()

        st.multiselect(
            "只在选中的文档中检索（不选 = 全部）",
            options=all_files,
            key="selected_files",
        )
    else:
        st.caption("暂无已索引的文档")


# ================= 初始化聊天引擎 =================
current_state = (
    tuple(sorted(st.session_state.get("selected_files", []))),
    st.session_state.get("index_version", 0),
    st.session_state.get("current_kb", "default"),
    st.session_state.get("date_range", "全部"),
)
if st.session_state.get("chat_state") != current_state:
    st.session_state.pop("chat_engine", None)
    st.session_state.chat_state = current_state

if "chat_engine" not in st.session_state:
    selected_files = st.session_state.get("selected_files", [])
    date_range = st.session_state.get("date_range", "全部")
    retriever = build_retriever(
        index, hybrid_retriever, _kb_name, selected_files, date_range
    )
    st.session_state.chat_engine = build_chat_engine(
        retriever, rerank, st.session_state.messages
    )


# ================= 聊天逻辑 =================
if prompt := st.chat_input(cfg.get("chat_placeholder", "请输入你的问题")):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt is not None and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("检索并生成中..."):
            response = st.session_state.chat_engine.chat(prompt)
            full_text = response.response
            source_nodes = response.source_nodes or []

            # 从 Chroma 反查补齐 metadata（修复向量分支 node 的 file_name 缺失）
            source_nodes = backfill_source_metadata(source_nodes, _kb_name)

            # ---- 去掉 LLM 末尾自己加的纯编号行，比如 [1][1][2] ----
            full_text = strip_trailing_citations(full_text)

            # ---- 提取回答里出现的引用编号，按数字排序 ----
            cited_nums = sorted(set(
                int(x) for x in re.findall(r"\[(\d+)\]", full_text)
            ))

            # ---- 构造引用行：[1] 文件名 / [1] 文件名 第 3 页 ----
            refs_lines = []
            for n in cited_nums:
                if 1 <= n <= len(source_nodes):
                    node = source_nodes[n - 1]
                    fn = node.metadata.get("file_name")
                    if not fn:
                        fp = node.metadata.get("file_path")
                        if fp:
                            fn = os.path.basename(fp)
                    if not fn:
                        fn = "未知文件"
                    pg = node.metadata.get("page", "?")
                    if pg in ("-", "?", None):
                        refs_lines.append(f"[{n}] {fn}")
                    else:
                        refs_lines.append(f"[{n}] {fn} 第 {pg} 页")

            # ---- 渲染正文 ----
            st.markdown(full_text)

            # ---- 空一行 + 引用列表 ----
            if refs_lines:
                st.markdown("")
                st.markdown("  \n".join(refs_lines))

            # ---- 侧边栏：折叠显示原文 ----
            with st.sidebar:
                st.subheader("检索到的原文")
                if not source_nodes:
                    st.info("未检索到相关片段。")
                else:
                    for i, node in enumerate(source_nodes, 1):
                        score = f"{node.score:.3f}" if node.score is not None else "N/A"

                        # 文件名兜底
                        fn = node.metadata.get("file_name")
                        if not fn:
                            fp = node.metadata.get("file_path")
                            if fp:
                                fn = os.path.basename(fp)
                        if not fn:
                            fn = "未知文件"

                        pg = node.metadata.get("page", "?")
                        if pg in ("-", "?", None):
                            title = f"[{i}] {fn}（分数：{score}）"
                        else:
                            title = f"[{i}] {fn} 第 {pg} 页（分数：{score}）"

                        # 折叠面板：默认收起
                        with st.expander(title, expanded=False):
                            src = node.metadata.get("source", "")
                            dt = node.metadata.get("date", "")
                            if src or dt:
                                st.caption(f"来源：{src or '?'}｜日期：{dt or '?'}")

                            preview = node.text[:2000] + ("..." if len(node.text) > 2000 else "")
                            st.markdown(f"```\n{preview}\n```")

            st.session_state.messages.append(
                {"role": "assistant", "content": full_text}
            )