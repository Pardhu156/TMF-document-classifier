"""Reusable file text-extraction utilities for uploaded TMF documents."""

from __future__ import annotations

from pathlib import Path


def extract_text_from_pdf(file_path: Path | str) -> str:
    """Extract text from a PDF using PyMuPDF, with PyPDF2 as a fallback."""
    path = Path(file_path)
    try:
        import fitz  # PyMuPDF

        with fitz.open(path) as document:
            return "\n".join(page.get_text() for page in document)
    except Exception:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_docx(file_path: Path | str) -> str:
    """Extract paragraph text from a DOCX file."""
    from docx import Document

    document = Document(str(file_path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def extract_text_from_txt(file_path: Path | str) -> str:
    """Extract text from a UTF-8-ish TXT file, ignoring invalid bytes."""
    return Path(file_path).read_text(encoding="utf-8", errors="ignore")
