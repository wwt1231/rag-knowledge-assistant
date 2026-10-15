"""配置加载、路径管理、日志设置"""

import os
import logging
from logging.handlers import RotatingFileHandler

import yaml


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 默认读 config.yaml；如果设置了 RAG_CONFIG，就用指定的配置文件
_config_name = os.environ.get("RAG_CONFIG", "config.yaml")
if not os.path.isabs(_config_name):
    CONFIG_PATH = os.path.join(BASE_DIR, _config_name)
else:
    CONFIG_PATH = _config_name


DEFAULT_CONFIG = {
    "chroma_dir": "chroma_db",
    "image_dir": "extracted_images",
    "knowledge_bases": {
        "default": {
            "name": "📚 默认知识库",
            "collection_name": "default_kb",
            "data_dir": "data/default",
            "indexed_files_path": "indexed_files_default.json",
        },
    },
    "llm": {
        "model": "qwen2.5:7b",
        "base_url": "http://localhost:11434",
        "temperature": 0.1,
        "num_ctx": 8192,
        "num_predict": 1024,
    },
    "embedding": {
        "model": "bge-m3",
        "base_url": "http://localhost:11434",
    },
    "retrieval": {
        "similarity_top_k": 10,
        "num_queries": 3,
        "use_async": True,
        "rerank_model": "BAAI/bge-reranker-base",
        "rerank_top_n": 5,
    },
    "system_prompt": "你是一个严谨的知识助手，只根据提供的资料回答，并给出出处。",
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        print(f"[警告] 未找到 {CONFIG_PATH}，使用默认配置。")
        return dict(DEFAULT_CONFIG)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return _deep_merge(DEFAULT_CONFIG, user_cfg)


cfg = load_config()


def _abs_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)


CHROMA_DIR = _abs_path(cfg["chroma_dir"])
IMAGE_DIR = _abs_path(cfg.get("image_dir", "extracted_images"))


def get_kb_config(kb_name: str) -> dict:
    kbs = cfg.get("knowledge_bases", {})
    if kb_name not in kbs:
        kb_name = list(kbs.keys())[0] if kbs else "default"
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


def ensure_dirs():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    for kb_name in cfg.get("knowledge_bases", {}).keys():
        kb = get_kb_config(kb_name)
        os.makedirs(kb["data_dir"], exist_ok=True)


LOG_PATH = os.path.join(BASE_DIR, "app.log")

logger = logging.getLogger("rag")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    file_handler = RotatingFileHandler(
        LOG_PATH, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))
    logger.addHandler(console_handler)