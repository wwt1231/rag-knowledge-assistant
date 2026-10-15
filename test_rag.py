"""
RAG 端到端功能测试集
用法：
    conda activate myenv
    cd D:\ai_project\assistant
    python test_rag.py
    python test_rag.py --kb physics     # 只测某个知识库
"""

import os
import sys
import re
import json
import csv
import argparse
from datetime import datetime

import yaml
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core import PromptTemplate
from llama_index.core.schema import TextNode

try:
    import jieba
    def chinese_tokenizer(text: str):
        return list(jieba.cut_for_search(text))
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False


# ================= 颜色 =================
class C:
    OK = "\033[92m"
    WARN = "\033[93m"
    ERR = "\033[91m"
    INFO = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def cprint(text, color=C.END):
    print(f"{color}{text}{C.END}")


# ================= 路径 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
QUESTIONS_PATH = os.path.join(BASE_DIR, "test_questions.json")


def _abs_path(p):
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)


# ================= 加载配置 =================
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_kb_config(cfg, kb_name):
    kbs = cfg.get("knowledge_bases", {})
    kb = kbs.get(kb_name, {})
    return {
        "key": kb_name,
        "name": kb.get("name", kb_name),
        "collection_name": kb.get("collection_name", f"{kb_name}_kb"),
        "data_dir": _abs_path(kb.get("data_dir", f"data/{kb_name}")),
    }


# ================= 构建 RAG =================
QA_PROMPT = PromptTemplate(
    "以下是检索到的资料片段：\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "请只根据以上资料回答：{query_str}\n\n"
    "要求：\n"
    "1. 用中文回答\n"
    "2. 引用时用 [编号] 标注\n"
    "3. 资料里提到了相关内容就必须回答，不要因为'资料没明确说'就拒答\n"
    "4. 只有资料里完全找不到相关信息时，才回答'根据现有资料无法回答'\n"
    "5. 数字、术语、单位忠于原文，不要改动\n"
)


def build_query_engine(cfg, kb_name, chroma_dir, rerank):
    """为指定知识库构建 query_engine。"""
    Settings.llm = Ollama(
        model=cfg["llm"]["model"],
        base_url=cfg["llm"]["base_url"],
        temperature=cfg["llm"]["temperature"],
        additional_kwargs={
            "num_ctx": cfg["llm"]["num_ctx"],
            "num_predict": cfg["llm"]["num_predict"],
        },
        system_prompt=cfg["system_prompt"],
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=cfg["embedding"]["model"],
        base_url=cfg["embedding"]["base_url"],
    )

    kb = get_kb_config(cfg, kb_name)
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(kb["collection_name"])

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )

    # 向量检索
    vector_retriever = index.as_retriever(
        similarity_top_k=cfg["retrieval"]["similarity_top_k"],
    )

    # BM25 检索
    result = collection.get(include=["documents", "metadatas"])
    nodes = []
    if result["documents"]:
        for i, text in enumerate(result["documents"]):
            meta = result["metadatas"][i] if result["metadatas"] else {}
            nodes.append(TextNode(text=text, metadata=dict(meta) if meta else {}))

    if nodes and _HAS_JIEBA:
        bm25 = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=cfg["retrieval"]["similarity_top_k"],
            tokenizer=chinese_tokenizer,
        )
        retriever = QueryFusionRetriever(
            [vector_retriever, bm25],
            similarity_top_k=cfg["retrieval"]["similarity_top_k"],
            num_queries=cfg["retrieval"].get("num_queries", 1),
            mode="reciprocal_rerank",
            use_async=False,
            verbose=False,
        )
    else:
        retriever = vector_retriever

    response_synthesizer = get_response_synthesizer(text_qa_template=QA_PROMPT)
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[rerank],
    )
    return query_engine


# ================= 文本归一化（用于宽松匹配） =================
def _normalize(text: str) -> str:
    """去掉所有空白字符，并把全角数字转半角，用于宽松匹配。"""
    if not text:
        return ""
    # 去空白
    text = re.sub(r"\s+", "", text)
    # 全角数字 → 半角
    full = "０１２３４５６７８９"
    half = "0123456789"
    for f, h in zip(full, half):
        text = text.replace(f, h)
    return text


# ================= 拒答判断 =================
def is_refused(answer: str, kw_hit: int = 0) -> bool:
    """判断 LLM 是不是拒答了。
    规则：
    1. 空答案 = 拒答
    2. 命中关键词 = 不算拒答（即使前面说了"无法回答"）
    3. 只看开头 30 字符，避免"前面拒答、后面又答"被误判
    """
    if not answer:
        return True
    if kw_hit > 0:
        return False
    head = answer.strip()[:30]
    patterns = [
        "无法回答", "没有找到", "未提及", "没有相关",
        "资料里没有", "现有资料", "无法确定",
        "没有提供", "不包含", "未包含",
    ]
    return any(p in head for p in patterns)


# ================= 关键词命中（忽略空格） =================
def keyword_hit(answer: str, keywords: list) -> tuple:
    """返回 (命中数, 总数)。匹配时忽略空格、全角转半角。"""
    if not keywords:
        return 0, 0
    norm_answer = _normalize(answer)
    hit = sum(1 for k in keywords if _normalize(k) in norm_answer)
    return hit, len(keywords)


def context_hit(context: str, keywords: list) -> tuple:
    """返回 (命中数, 总数)。匹配时忽略空格、全角转半角。"""
    if not keywords:
        return 0, 0
    norm_ctx = _normalize(context)
    hit = sum(1 for k in keywords if _normalize(k) in norm_ctx)
    return hit, len(keywords)


# ================= 单条测试 =================
def run_one(qe, q: dict, kb_name: str) -> dict:
    """跑一条问题，返回结果字典。"""
    question = q["question"]
    keywords = q.get("keywords") or []
    answerable = q.get("answerable", True)

    result = {
        "id": q["id"],
        "kb": kb_name,
        "category": q.get("category", ""),
        "question": question,
        "answerable": answerable,
        "expected_keywords": "、".join(keywords),
        "answer": "",
        "refused": False,
        "refusal_correct": None,
        "kw_hit_answer": 0,
        "kw_total": len(keywords),
        "kw_hit_context": 0,
        "n_nodes": 0,
        "top_file": "",
        "error": "",
    }

    try:
        resp = qe.query(question)
        answer = resp.response or ""
        nodes = resp.source_nodes or []
        context = "\n".join(n.text for n in nodes)

        result["answer"] = answer[:200]
        result["n_nodes"] = len(nodes)
        if nodes:
            result["top_file"] = nodes[0].metadata.get("file_name", "")

        # === 先算关键词命中，再判断是否拒答 ===
        hit_a, _ = keyword_hit(answer, keywords)
        hit_c, _ = context_hit(context, keywords)
        result["kw_hit_answer"] = hit_a
        result["kw_hit_context"] = hit_c

        refused = is_refused(answer, kw_hit=hit_a)
        result["refused"] = refused
        result["refusal_correct"] = (not refused) if answerable else refused

    except Exception as e:
        result["error"] = str(e)[:200]

    return result


# ================= 汇总 =================
def summarize(results: list, kb_name: str):
    total = len(results)
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

    err = [r for r in results if r["error"]]
    correct_refusal = [r for r in unanswerable if r["refusal_correct"]]
    false_refusal = [r for r in answerable if r["refusal_correct"] is False]

    kw_ans = sum(r["kw_hit_answer"] for r in answerable)
    kw_total = sum(r["kw_total"] for r in answerable)
    kw_ctx = sum(r["kw_hit_context"] for r in answerable)

    cprint(f"\n{'=' * 60}", C.BOLD)
    cprint(f"📊 「{kb_name}」测试汇总", C.BOLD + C.INFO)
    cprint(f"{'=' * 60}", C.BOLD)

    print(f"  总问题数              : {total}")
    print(f"  可回答                : {len(answerable)}")
    print(f"  不可回答              : {len(unanswerable)}")
    if err:
        cprint(f"  执行错误              : {len(err)}", C.ERR)
    print()

    if answerable:
        rate = 1 - len(false_refusal) / len(answerable)
        color = C.OK if rate >= 0.9 else (C.WARN if rate >= 0.7 else C.ERR)
        cprint(f"  可回答题正确应答率    : {rate:.1%}  ({len(answerable) - len(false_refusal)}/{len(answerable)})", color)

    if unanswerable:
        rate = len(correct_refusal) / len(unanswerable)
        color = C.OK if rate >= 0.9 else (C.WARN if rate >= 0.7 else C.ERR)
        cprint(f"  拒答题正确拒答率      : {rate:.1%}  ({len(correct_refusal)}/{len(unanswerable)})", color)

    if kw_total:
        rate = kw_ans / kw_total
        color = C.OK if rate >= 0.6 else (C.WARN if rate >= 0.4 else C.ERR)
        cprint(f"  答案关键词命中率      : {rate:.1%}  ({kw_ans}/{kw_total})", color)

    if kw_total:
        rate = kw_ctx / kw_total
        color = C.OK if rate >= 0.8 else (C.WARN if rate >= 0.6 else C.ERR)
        cprint(f"  检索关键词命中率      : {rate:.1%}  ({kw_ctx}/{kw_total})", color)

    # 改进建议
    cprint(f"\n  💡 改进建议：", C.INFO)
    if unanswerable and len(correct_refusal) / len(unanswerable) < 0.8:
        cprint(f"     - 拒答率低 → system_prompt 再强调'资料里没有就直说'", C.WARN)
    if answerable and len(false_refusal) / len(answerable) > 0.2:
        cprint(f"     - 误拒答多 → 检查 chunk_size 是否太大，把答案切散", C.WARN)
    if kw_total and kw_ctx / kw_total < 0.6:
        cprint(f"     - 检索命中低 → 增加 top_k、调 chunk_size、考虑换 embedding 模型", C.WARN)
    if kw_total and kw_ans / kw_total < 0.4 and kw_ctx / kw_total >= 0.6:
        cprint(f"     - 检索准但答案差 → prompt 或 LLM 生成有问题", C.WARN)


# ================= 主流程 =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", type=str, default=None,
                        help="只测指定知识库（默认全部）")
    args = parser.parse_args()

    cprint("🧪 RAG 端到端测试", C.BOLD + C.INFO)

    cfg = load_config()
    chroma_dir = _abs_path(cfg.get("chroma_dir", "chroma_db"))

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # 加载 Rerank（全局共用）
    cprint("\n⏳ 加载 Rerank 模型...", C.INFO)
    rerank = SentenceTransformerRerank(
        model=cfg["retrieval"]["rerank_model"],
        top_n=cfg["retrieval"]["rerank_top_n"],
    )
    cprint("✅ Rerank 加载完成", C.OK)

    # 按 kb 分组
    by_kb = {}
    for q in questions:
        kb = q.get("kb", "default")
        by_kb.setdefault(kb, []).append(q)

    all_results = []

    for kb_name, qs in by_kb.items():
        if args.kb and kb_name != args.kb:
            continue

        # "any" 类问题分配给所有知识库？这里简单起见跳过
        if kb_name == "any":
            continue

        kb = get_kb_config(cfg, kb_name)
        cprint(f"\n{'=' * 60}", C.INFO)
        cprint(f"📚 测试「{kb['name']}」({len(qs)} 条)", C.BOLD + C.INFO)
        cprint(f"{'=' * 60}", C.INFO)

        try:
            qe = build_query_engine(cfg, kb_name, chroma_dir, rerank)
        except Exception as e:
            cprint(f"❌ 构建 query_engine 失败：{e}", C.ERR)
            continue

        for i, q in enumerate(qs, 1):
            print(f"  [{i}/{len(qs)}] {q['question'][:40]}...", end=" ")
            r = run_one(qe, q, kb_name)
            all_results.append(r)

            if r["error"]:
                cprint(f"❌ {r['error'][:50]}", C.ERR)
            elif not r["answerable"] and r["refusal_correct"]:
                cprint("✅ 正确拒答", C.OK)
            elif not r["answerable"] and not r["refusal_correct"]:
                cprint("❌ 应该拒答但没拒", C.ERR)
            elif r["answerable"] and r["refusal_correct"]:
                hit = r["kw_hit_answer"]
                total = r["kw_total"]
                if total and hit == total:
                    cprint(f"✅ 命中 {hit}/{total}", C.OK)
                elif total and hit > 0:
                    cprint(f"⚠️  命中 {hit}/{total}", C.WARN)
                else:
                    cprint(f"❌ 命中 0/{total}", C.ERR)
            else:
                cprint("⚠️  该答没答", C.WARN)

    if not all_results:
        cprint("\n⚠️ 没有测试到任何问题（可能 --kb 参数不对）", C.WARN)
        return

    # 汇总
    summarize(all_results, args.kb or "全部")

    # CSV
    csv_path = os.path.join(
        BASE_DIR, f"rag_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    cprint(f"\n📄 明细已保存：{csv_path}", C.OK)


if __name__ == "__main__":
    main()