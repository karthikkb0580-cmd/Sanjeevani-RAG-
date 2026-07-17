"""app/loaders/__init__.py – Document Loader Factory"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.loaders.pdf_loader import PDFLoader
from app.loaders.txt_loader import TXTLoader
from app.loaders.docx_loader import DOCXLoader

logger = logging.getLogger(__name__)

_LOADER_MAP = {
    ".pdf": PDFLoader,
    ".txt": TXTLoader,
    ".docx": DOCXLoader,
    ".doc": DOCXLoader,
}


def load_document(file_path: str | Path) -> dict[str, Any]:
    """
    Load a document from disk using the appropriate loader.

    Returns a standardised document dict:
    {
        "title": str,
        "author": str,
        "file_path": str,
        "total_pages": int,
        "metadata": dict,
        "pages": [{"page": int, "text": str, "headings": [...], "section": str}]
    }

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    loader_cls = _LOADER_MAP.get(ext)
    if loader_cls is None:
        supported = ", ".join(_LOADER_MAP.keys())
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported types: {supported}"
        )

    loader = loader_cls(path)
    document = loader.load()
    logger.info(
        "Loaded document '%s' – %d page(s)",
        document.get("title", path.stem),
        document.get("total_pages", 0),
    )
    return document


__all__ = ["PDFLoader", "TXTLoader", "DOCXLoader", "load_document"]
