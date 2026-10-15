"""常量与文件解析器映射"""

from llama_index.readers.file import (
    DocxReader,
    FlatReader,
    HTMLTagReader,
    PandasCSVReader,
    PandasExcelReader,
    PptxReader,
)


SUPPORTED_EXTENSIONS = [
    ".pdf", ".docx", ".pptx",
    ".txt", ".md", ".json",
    ".csv", ".xlsx", ".xls",
    ".html", ".htm",
]


EXCLUDED_META_KEYS = [
    "creation_date",
    "last_modified_date",
    "file_path",
    "images",
    "file_size",
    "file_type",
    "date_ts",
    "url",
]


META_KEYS = (
    "file_name", "file_path", "page",
    "date", "date_ts", "source", "url", "title", "category",
)


_FILE_EXTRACTOR = {
    ".docx": DocxReader(),
    ".pptx": PptxReader(),
    ".csv":  PandasCSVReader(),
    ".xlsx": PandasExcelReader(),
    ".xls":  PandasExcelReader(),
    ".json": FlatReader(),
    ".txt":  FlatReader(),
    ".md":   FlatReader(),
    ".html": HTMLTagReader(),
    ".htm":  HTMLTagReader(),
}


def build_file_extractor():
    return _FILE_EXTRACTOR