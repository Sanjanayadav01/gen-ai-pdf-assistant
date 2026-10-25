"""PDF loading: extract text page by page so every chunk can cite its page number."""
from dataclasses import dataclass
from typing import BinaryIO, Union

from pypdf import PdfReader


@dataclass
class PageText:
    source: str
    page: int  # 1-based
    text: str


def load_pdf(file: Union[str, BinaryIO], name: str) -> list[PageText]:
    """Return the non-empty pages of a PDF. `file` may be a path or a file-like object."""
    reader = PdfReader(file)
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"'{name}' is password protected.") from exc

    pages: list[PageText] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(PageText(source=name, page=i, text=text))

    if not pages:
        raise ValueError(
            f"No extractable text in '{name}'. It may be a scanned PDF (needs OCR)."
        )
    return pages
