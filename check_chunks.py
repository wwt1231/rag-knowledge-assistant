import chromadb
import json

client = chromadb.PersistentClient(path="chroma_db")
col = client.get_collection("port_kb")

print(f"总 chunk 数：{col.count()}\n")

# ===== 1. 看第一个 chunk 的完整 relationships 结构 =====
sample = col.get(include=["metadatas"], limit=1)
m = sample["metadatas"][0]
nc = m.get("_node_content", "")
print("=== 完整 _node_content ===")
print(nc[:2000])
print()

# ===== 2. 关键词搜索 =====
print("=== 关键词搜索（42 个 chunk 里有几个提到） ===")
for kw in ["塑限", "液限", "30.6", "58", "氯离子", "Cl-", "可溶性盐", "0.00", "wP", "wL"]:
    try:
        r = col.get(include=["documents"], where_document={"$contains": kw})
        print(f"  '{kw}': {len(r['documents'])} 个 chunk")
    except Exception as e:
        print(f"  '{kw}': 出错 {e}")