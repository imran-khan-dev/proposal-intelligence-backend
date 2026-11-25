"""Minimal PDF generator for test fixtures.

Only pypdf (a reader) is available in this environment, so test PDFs are
constructed from raw PDF syntax.  Each page emits its text lines with
``Td``/``Tj`` operators so pypdf can extract them page by page.
"""

from __future__ import annotations

from typing import List


def _pdf_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _content_stream(lines: List[str]) -> bytes:
    """Build a PDF content stream that shows each text line on its own row."""
    ops = ["BT", "/F1 12 Tf", "50 740 Td"]
    y = 740
    for line in lines:
        ops.append(f"({_pdf_escape(line)}) Tj")
        y -= 14
        ops.append("0 -14 Td")
    ops.append("ET")
    stream = "\n".join(ops) + "\n"
    return stream.encode("latin-1")


def build_pdf(pages: List[List[str]]) -> bytes:
    """Build a minimal valid multi-page PDF.

    Args:
        pages: A list of pages; each page is a list of text lines.

    Returns:
        The PDF as bytes.
    """
    num_pages = len(pages)

    # Object index assignment.
    # 1: Catalog, 2: Pages, then per page: Page, Content, then Font last.
    obj_offsets = {}
    parts: List[bytes] = [b"%PDF-1.4\n% test fixture\n"]

    def obj(number: int, content: bytes) -> None:
        obj_offsets[number] = len(b"".join(parts))
        parts.append(f"{number} 0 obj\n".encode("ascii") + content + b"\nendobj\n")

    # Font object (object 3) -- assigned early so numbering is stable.
    # We'll number: 1 Catalog, 2 Pages, 3 Font, then pages/contents.
    font_obj_num = 3
    obj(
        font_obj_num,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    )

    # Build pages and content streams.
    page_obj_nums: List[int] = []
    content_obj_nums: List[int] = []
    next_num = font_obj_num + 1
    for _ in pages:
        page_obj_nums.append(next_num)
        next_num += 1
        content_obj_nums.append(next_num)
        next_num += 1

    catalog_num = 1
    pages_num = 2
    # Reorder: catalog=1, pages=2, font=3 already done. Fix font to be after pages?
    # PDF doesn't require a particular order; xref references by object number.
    # Keep as assigned.

    # Catalog
    obj(catalog_num, f"<< /Type /Catalog /Pages {pages_num} 0 R >>".encode("ascii"))
    # Pages
    kids = " ".join(f"{p} 0 R" for p in page_obj_nums)
    obj(
        pages_num,
        f"<< /Type /Pages /Kids [{kids}] /Count {num_pages} >>".encode("ascii"),
    )
    # Font was already written as object 3; that's fine (order irrelevant).

    # Pages + content streams
    for i, (page_num, content_num, lines) in enumerate(
        zip(page_obj_nums, content_obj_nums, pages)
    ):
        stream = _content_stream(lines)
        obj(
            content_num,
            b"<< /Length " + str(len(stream)).encode("ascii") + b">>\nstream\n"
            + stream
            + b"endstream",
        )
        obj(
            page_num,
            (
                f"<< /Type /Page /Parent {pages_num} 0 R "
                f"/MediaBox [0 0 612 792] /Contents {content_num} 0 R "
                f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> >>"
            ).encode("ascii"),
        )

    # Build xref
    xref_offset = len(b"".join(parts))
    xref_lines = [b"xref", f"0 {next_num} 0 obj".encode()]  # placeholder
    # Actually xref header: "xref\n0 <count>\n"
    xref_body = [f"0 {next_num}\n".encode("ascii")]
    # Free object 0
    xref_body.append(b"0000000000 65535 f \n")
    for n in range(1, next_num):
        off = obj_offsets.get(n, 0)
        xref_body.append(f"{off:010d} 00000 n \n".encode("ascii"))

    parts.append(b"xref\n")
    parts.append(b"".join(xref_body))
    parts.append(b"trailer\n")
    parts.append(f"<< /Size {next_num} /Root {catalog_num} 0 R >>\n".encode("ascii"))
    parts.append(b"startxref\n")
    parts.append(f"{xref_offset}\n".encode("ascii"))
    parts.append(b"%%EOF\n")

    return b"".join(parts)
