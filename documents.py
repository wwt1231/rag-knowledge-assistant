"""文档解析 + metadata 提取"""

import os
import re
import time as _time

import pymupdf4llm
from llama_index.core import SimpleDirectoryReader, Document

from constants import (
    SUPPORTED_EXTENSIONS,
    EXCLUDED_META_KEYS,
    build_file_extractor,
)


# ================= PDF 专用解析 =================
def load_pdf_with_pymupdf4llm(file_path: str) -> list:
    fname = os.path.basename(file_path)

    pages = pymupdf4llm.to_markdown(
        file_path,
        write_images=False,
        page_chunks=True,
    )

    docs = []
    for page in pages:
        if isinstance(page, dict):
            raw_text = page.get("text", "") or ""
            page_meta = page.get("metadata", {}) or {}
        else:
            raw_text = str(page)
            page_meta = {}

        if not raw_text.strip():
            continue

        page_num = int(page_meta.get("page", 0)) + 1

        doc = Document(
            text=raw_text,
            metadata={
                "file_path": str(file_path),
                "file_name": fname,
                "page": page_num,
                "images": "[]",
                "date_ts": 0,
            },
        )
        doc.excluded_embed_metadata_keys = EXCLUDED_META_KEYS
        doc.excluded_llm_metadata_keys = EXCLUDED_META_KEYS
        docs.append(doc)

    if not docs:
        raise ValueError(f"PDF 解析失败，未提取到任何内容：{fname}")
    return docs


# ================= 非 PDF 解析 =================
def _parse_frontmatter(text: str) -> tuple:
    text = text.lstrip("\ufeff")
    m = re.match(r"^\ufeff?\s*---[ \t]*\n(.*?)\n---[ \t]*\n", text, re.DOTALL)
    if not m:
        return {}, text

    fm = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip().strip('"').strip("'")
        fm[k.strip()] = v

    return fm, text[m.end():]


def load_other_document(file_path: str) -> list:
    reader = SimpleDirectoryReader(
        input_files=[file_path],
        file_extractor=build_file_extractor(),
    )
    docs = reader.load_data()

    fname = os.path.basename(file_path)
    for doc in docs:
        fm, body = _parse_frontmatter(doc.text)
        if fm:
            doc.text = body

        meta = {
            "file_name": fname,
            "file_path": file_path,
            "page": "-",
            "images": "[]",
        }

        if fm.get("date"):
            meta["date"] = fm["date"]
            try:
                meta["date_ts"] = int(
                    _time.mktime(_time.strptime(fm["date"], "%Y-%m-%d"))
                )
            except Exception:
                meta["date_ts"] = 0
        if fm.get("source"):
            meta["source"] = fm["source"]
        if fm.get("url"):
            meta["url"] = fm["url"]
        if fm.get("title"):
            meta["title"] = fm["title"]
        if fm.get("category"):
            meta["category"] = fm["category"]

        doc.metadata = meta
        doc.excluded_embed_metadata_keys = EXCLUDED_META_KEYS
        doc.excluded_llm_metadata_keys = EXCLUDED_META_KEYS
    return docs


# ================= 统一入口 =================
def load_any_document(file_path: str) -> list:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf_with_pymupdf4llm(file_path)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件类型：{ext}")
    return load_other_document(file_path)