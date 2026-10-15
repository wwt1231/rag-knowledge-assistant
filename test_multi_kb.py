"""
多知识库功能测试脚本
用途：在命令行快速验证 config.yaml 和各个知识库的状态

用法：
    conda activate myenv
    cd D:\ai_project\assistant
    python test_multi_kb.py
"""

import os
import json
import yaml
import chromadb
from pathlib import Path


# ================= 颜色输出（Windows 也能用） =================
class C:
    OK = "\033[92m"       # 绿
    WARN = "\033[93m"     # 黄
    ERR = "\033[91m"      # 红
    INFO = "\033[96m"     # 青
    BOLD = "\033[1m"
    END = "\033[0m"


def cprint(text, color=C.END):
    print(f"{color}{text}{C.END}")


# ================= 路径基准 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")


def load_config():
    if not os.path.exists(CONFIG_PATH):
        cprint(f"❌ 找不到 config.yaml：{CONFIG_PATH}", C.ERR)
        raise SystemExit(1)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _abs_path(p):
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)


def get_kb_config(cfg, kb_name):
    kbs = cfg.get("knowledge_bases", {})
    kb = kbs.get(kb_name, {})
    return {
        "key": kb_name,
        "name": kb.get("name", kb_name),
        "collection_name": kb.get("collection_name", f"{kb_name}_kb"),
        "data_dir": _abs_path(kb.get("data_dir", f"data/{kb_name}")),
        "indexed_files_path": _abs_path(
            kb.get("indexed_files_path", f"indexed_files_{kb_name}.json")
        ),
    }


# ================= 单项检查 =================
def check_data_dir(kb):
    """检查数据目录和文件数。"""
    d = kb["data_dir"]
    if not os.path.exists(d):
        return False, f"目录不存在：{d}"

    files = [
        p for p in Path(d).rglob("*")
        if p.is_file()
        and not p.name.startswith(".")
        and not p.name.startswith("~$")
    ]
    return True, f"{len(files)} 个文件"


def check_collection(cfg, kb):
    """连接 Chroma，检查 collection。"""
    chroma_dir = _abs_path(cfg.get("chroma_dir", "chroma_db"))
    if not os.path.exists(chroma_dir):
        return False, 0, f"Chroma 目录不存在：{chroma_dir}"

    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        try:
            collection = client.get_collection(kb["collection_name"])
        except Exception:
            return True, 0, "collection 还没创建（正常，第一次用）"

        count = collection.count()
        return True, count, f"节点数 {count}"
    except Exception as e:
        return False, 0, f"连接失败：{e}"


def check_indexed_files(kb):
    """检查 indexed_files.json。"""
    p = kb["indexed_files_path"]
    if not os.path.exists(p):
        return True, 0, "记录文件不存在（正常，第一次用）"
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        n = len(data) if isinstance(data, list) else 0
        return True, n, f"记录 {n} 个文件"
    except Exception as e:
        return False, 0, f"解析失败：{e}"


def check_available_files(cfg, kb):
    """从 Chroma metadata 提取文件名（模拟 app.py 里的逻辑）。"""
    chroma_dir = _abs_path(cfg.get("chroma_dir", "chroma_db"))
    if not os.path.exists(chroma_dir):
        return []

    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        try:
            collection = client.get_collection(kb["collection_name"])
        except Exception:
            return []

        result = collection.get(include=["metadatas"])
        files = set()
        for meta in (result.get("metadatas") or []):
            if not meta:
                continue
            fn = None
            # 顶层
            if meta.get("file_name"):
                fn = meta["file_name"]
            # _node_content
            elif meta.get("_node_content"):
                try:
                    node = json.loads(meta["_node_content"])
                    rels = node.get("relationships", {}) or {}
                    for rel in rels.values():
                        rm = rel.get("metadata") or {}
                        if rm.get("file_name"):
                            fn = rm["file_name"]
                            break
                except Exception:
                    pass
            if fn:
                files.add(fn)
        return sorted(files)
    except Exception:
        return []


# ================= 单个知识库的报告 =================
def report_kb(cfg, kb_name):
    kb = get_kb_config(cfg, kb_name)
    cprint(f"\n{'=' * 60}", C.INFO)
    cprint(f"📚 {kb['name']}  (key={kb_name})", C.BOLD + C.INFO)
    cprint(f"{'=' * 60}", C.INFO)

    # 1. 配置
    print(f"  collection : {kb['collection_name']}")
    print(f"  data_dir   : {kb['data_dir']}")
    print(f"  index_file : {kb['indexed_files_path']}")

    # 2. 目录
    ok, msg = check_data_dir(kb)
    if ok:
        cprint(f"  ✅ 目录     : {msg}", C.OK)
    else:
        cprint(f"  ❌ 目录     : {msg}", C.ERR)

    # 3. Collection
    ok, count, msg = check_collection(cfg, kb)
    if ok:
        cprint(f"  ✅ Chroma   : {msg}", C.OK if count > 0 else C.WARN)
    else:
        cprint(f"  ❌ Chroma   : {msg}", C.ERR)

    # 4. Indexed files
    ok, n, msg = check_indexed_files(kb)
    if ok:
        cprint(f"  ✅ 索引记录 : {msg}", C.OK if n > 0 else C.WARN)
    else:
        cprint(f"  ❌ 索引记录 : {msg}", C.ERR)

    # 5. 文件列表
    files = check_available_files(cfg, kb)
    if files:
        cprint(f"  ✅ 已入库文件 ({len(files)}):", C.OK)
        for f in files[:5]:
            print(f"       - {f}")
        if len(files) > 5:
            print(f"       ... 还有 {len(files) - 5} 个")
    else:
        cprint(f"  ⚠️  已入库文件 : 空", C.WARN)

    # 6. 一致性检查
    if count > 0 and n == 0:
        cprint(f"  ⚠️  向量库有数据但 indexed_files 丢失（可自动重建）", C.WARN)


# ================= 汇总 =================
def summary(cfg):
    cprint(f"\n{'=' * 60}", C.BOLD)
    cprint(f"📊 汇总", C.BOLD)
    cprint(f"{'=' * 60}", C.BOLD)

    kbs = cfg.get("knowledge_bases", {})
    if not kbs:
        cprint("  ❌ config.yaml 里没有 knowledge_bases 配置", C.ERR)
        return

    print(f"  {'知识库':<20} {'节点数':>10} {'文件数':>10}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 10}")

    for kb_name in kbs.keys():
        kb = get_kb_config(cfg, kb_name)
        _, count, _ = check_collection(cfg, kb)
        files = check_available_files(cfg, kb)
        print(f"  {kb['name']:<20} {count:>10} {len(files):>10}")


# ================= 主流程 =================
def main():
    cprint("🔍 多知识库测试工具", C.BOLD + C.INFO)
    cprint(f"项目目录：{BASE_DIR}", C.INFO)

    cfg = load_config()

    # 检查顶层配置
    cprint(f"\n{'=' * 60}", C.INFO)
    cprint("⚙️  顶层配置", C.BOLD + C.INFO)
    cprint(f"{'=' * 60}", C.INFO)

    chroma_dir = _abs_path(cfg.get("chroma_dir", "chroma_db"))
    print(f"  chroma_dir : {chroma_dir}")
    if os.path.exists(chroma_dir):
        cprint(f"  ✅ 存在", C.OK)
    else:
        cprint(f"  ⚠️  不存在（首次启动会自动创建）", C.WARN)

    kbs = cfg.get("knowledge_bases", {})
    print(f"  知识库数量 : {len(kbs)}")
    if not kbs:
        cprint("  ❌ 没有配置任何知识库，请检查 config.yaml", C.ERR)
        return

    # 逐个知识库报告
    for kb_name in kbs.keys():
        report_kb(cfg, kb_name)

    # 汇总
    summary(cfg)

    cprint(f"\n{'=' * 60}", C.OK + C.BOLD)
    cprint("✅ 测试完成", C.OK + C.BOLD)
    cprint(f"{'=' * 60}", C.OK + C.BOLD)


if __name__ == "__main__":
    main()