import chromadb
import json

client = chromadb.PersistentClient(path="chroma_db")

# 换成你当前的知识库
KB = "port_kb"   # 或 "default_kb"

col = client.get_collection(KB)
print(f"=== {KB} 总数 {col.count()} ===\n")

# 拉前 20 条的 metadata
r = col.get(include=["metadatas"], limit=20)

missing = 0
for i, m in enumerate(r["metadatas"]):
    fn = m.get("file_name")
    fp = m.get("file_path")
    nc = m.get("_node_content", "")

    if not fn and not fp:
        missing += 1
        print(f"[{i}] ❌ file_name/file_path 都缺")
        print(f"      顶层 keys: {list(m.keys())}")
        if nc:
            try:
                d = json.loads(nc)
                print(f"      node.metadata: {d.get('metadata')}")
                rels = d.get("relationships", {})
                for k, v in rels.items():
                    print(f"      rel[{k}]: {v.get('metadata')}")
            except Exception as e:
                print(f"      解析失败: {e}")
    else:
        print(f"[{i}] ✅ {fn or fp}")

print(f"\n前 20 条里，缺失 {missing} 条")