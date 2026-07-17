"""tests/test_txt_loader.py – Unit tests for TXTLoader."""

from __future__ import annotations

import pytest
from pathlib import Path
from app.loaders.txt_loader import TXTLoader


def test_txt_loader_missing_file_raises_error():
    with pytest.raises(FileNotFoundError):
        TXTLoader("non_existent_file.txt")


def test_txt_loader_success(tmp_path: Path):
    # Create a temporary txt file
    txt_file = tmp_path / "test_doc.txt"
    content = (
        "My Awesome Research Document\n"
        "============================\n"
        "This is the introduction paragraph.\n"
        "\n"
        "1. Background Section\n"
        "Here is some background text about cell biology.\n"
        "It spans multiple lines.\n"
        "\n"
        "\x0c\n"  # Page break character
        "# Advanced Findings\n"
        "Here are the findings and other details.\n"
    )
    txt_file.write_text(content, encoding="utf-8")

    loader = TXTLoader(txt_file)
    data = loader.load()

    assert data["title"] == "My Awesome Research Document"
    assert data["total_pages"] >= 2
    assert "TXT" in data["metadata"]["format"]
    
    # Assert pages structure
    pages = data["pages"]
    assert pages[0]["page"] == 1
    assert "1. Background Section" in pages[0]["headings"]
    
    assert pages[1]["page"] == 2
    assert "# Advanced Findings" in pages[1]["headings"]
