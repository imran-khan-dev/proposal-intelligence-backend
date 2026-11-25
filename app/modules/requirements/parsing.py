"""Structured RFP parsing.

Turns per-page text segments (the same segments produced by the ingestion
pipeline and reconstructable from stored chunks) into a
:class:`~app.modules.requirements.structured_document.StructuredDocument`.

The parser is deliberately lightweight (no external framework).  It detects:

- numbered / markdown / all-caps headings and builds a section hierarchy
- numbered and bulleted list items
- pipe- or multi-space-delimited tables (rows and cells)
- paragraphs (consecutive non-empty lines)
- empty / poorly extracted pages (recorded as warnings)

Character offsets are absolute into the normalized document text, matching
the offsets stored on chunks during ingestion, so every parsed element and
extracted requirement remains traceable to its source.

Limitation: table detection relies on the extracted text preserving column
separators (pipes or 2+ spaces).  pypdf's default text extraction may collapse
spacing for real-world PDFs; such tables fall back to paragraphs.  No OCR is
performed in this milestone.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, Optional

from app.modules.requirements.structured_document import (
    DocumentElement,
    Section,
    SourceSpan,
    StructuredDocument,
    StructuredTable,
    TableCell,
)

# A numbered heading: "1. Title", "1.2 Scope", "3.2.1 Details".
_NUMBERED_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+(\S.*?)\s*$")
# A markdown heading: "# Title", "## Sub".
_MD_HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(\S.*?)\s*$")
# A list item: "- x", "* x", "1. x", "1) x", "a) x", "iv. x".
_LIST_ITEM_RE = re.compile(
    r"^(\s*)([-*\u2022]|\d+[.)]|[a-zA-Z][.)]|[ivxIVX]+[.)])\s+(\S.*?)\s*$"
)
# A table row: contains a pipe, or 2+ spaces separating non-space tokens.
_TABLE_PIPE_RE = re.compile(r"\|")
_TABLE_MULTISPACE_RE = re.compile(r"\S(?:.*?\S)?\s{2,}\S")

_MIN_PAGE_TEXT = 10  # below this a page is considered poorly extracted

# Modal verbs that signal a requirement sentence rather than a heading title.
_MODAL_HINT_RE = re.compile(
    r"\b(must|shall|should|may|might|required|will|can|could)\b",
    re.IGNORECASE,
)
_MAX_HEADING_WORDS = 8


class BaseDocumentParser(ABC):
    """Abstract parser producing a :class:`StructuredDocument`."""

    @abstractmethod
    def parse(
        self,
        document_id: str,
        filename: str,
        pages: List[dict],
    ) -> StructuredDocument:
        """Parse per-page segments into a structured document.

        Args:
            document_id: Stable document identifier.
            filename: Original source filename.
            pages: List of page dicts with ``page_number``, ``text``,
                ``character_start`` (absolute offset of the page text).
        """
        ...


def _iter_lines_with_offsets(text: str, base_offset: int):
    """Yield ``(line, abs_start, abs_end)`` for each line in ``text``.

    ``abs_start``/``abs_end`` are absolute offsets into the normalized
    document text (``base_offset`` is the page's absolute start).
    """
    pos = 0
    for raw in text.split("\n"):
        start = base_offset + pos
        end = start + len(raw)
        yield raw, start, end
        pos += len(raw) + 1  # account for the stripped "\n"


def _is_all_caps_heading(line: str) -> bool:
    stripped = line.strip()
    if not (3 <= len(stripped) <= 60):
        return False
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return False
    return stripped == stripped.upper()


def _heading_level_and_title(line: str) -> Optional[tuple[int, str]]:
    """Return ``(level, title)`` if ``line`` is a heading, else ``None``.

    Numbered lines are ambiguous: "2.1 Scope" is a heading while
    "2.1 The system must support SAML." is a requirement.  A numbered line is
    treated as a heading only when its title is short and carries no modal
    verb or sentence-ending punctuation.
    """
    m = _NUMBERED_HEADING_RE.match(line)
    if m:
        numbering = m.group(1)
        title = m.group(2).strip()
        level = numbering.count(".") + 1
        if _looks_like_heading_title(title):
            return level, title
        return None
    m = _MD_HEADING_RE.match(line)
    if m:
        title = m.group(2).strip()
        if _looks_like_heading_title(title):
            return len(m.group(1)), title
        return None
    if _is_all_caps_heading(line):
        return 1, line.strip()
    return None


def _looks_like_heading_title(title: str) -> bool:
    if not title:
        return False
    if title.endswith((".", "!", "?", ":", ";", ",")):
        return False
    if _MODAL_HINT_RE.search(title):
        return False
    return len(title.split()) <= _MAX_HEADING_WORDS


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _heading_level_and_title(line) or _LIST_ITEM_RE.match(line):
        return False
    if _TABLE_PIPE_RE.search(stripped):
        return True
    return bool(_TABLE_MULTISPACE_RE.search(stripped))


def _split_table_cells(line: str) -> List[str]:
    stripped = line.strip()
    if _TABLE_PIPE_RE.search(stripped):
        parts = stripped.split("|")
        # Drop leading/trailing empty fragments from surrounding pipes.
        if parts and parts[0].strip() == "":
            parts = parts[1:]
        if parts and parts[-1].strip() == "":
            parts = parts[:-1]
        return [p.strip() for p in parts]
    return [c.strip() for c in re.split(r"\s{2,}", stripped) if c.strip()]


class StructuredTextParser(BaseDocumentParser):
    """Rule-based parser for page text segments."""

    def parse(
        self,
        document_id: str,
        filename: str,
        pages: List[dict],
    ) -> StructuredDocument:
        elements: List[DocumentElement] = []
        warnings: List[str] = []
        section_stack: List[Section] = []
        top_sections: List[Section] = []
        order = 0

        for page in pages:
            page_number = page.get("page_number", 1)
            text = page.get("text", "") or ""
            base_offset = int(page.get("character_start", 0))

            if not text.strip():
                warnings.append(f"Page {page_number} has no extractable text")
                continue
            if len(text.strip()) < _MIN_PAGE_TEXT:
                warnings.append(
                    f"Page {page_number} has very little extractable text"
                )

            current_path: Optional[str] = None
            lines = list(_iter_lines_with_offsets(text, base_offset))
            i = 0
            while i < len(lines):
                line, l_start, l_end = lines[i]
                stripped = line.strip()

                if not stripped:
                    i += 1
                    continue

                heading = _heading_level_and_title(line)
                list_match = _LIST_ITEM_RE.match(line)

                # Disambiguate numbered lines: inside a list context (after a
                # list item or a lead-in paragraph ending with ":"), a numbered
                # line is a list item rather than a section heading.
                in_list_context = self._in_list_context(elements)
                treat_as_list = list_match is not None and (
                    in_list_context or heading is None
                )

                if treat_as_list:
                    marker = list_match.group(2)
                    content = list_match.group(3)
                    elements.append(
                        DocumentElement(
                            kind="list_item",
                            text=content,
                            span=SourceSpan(
                                page_number=page_number,
                                char_start=l_start,
                                char_end=l_end,
                            ),
                            order=order,
                            section_path=current_path,
                            list_marker=marker,
                        )
                    )
                    order += 1
                    i += 1
                    continue

                if heading is not None:
                    level, title = heading
                    path = self._push_section(
                        section_stack, top_sections, level, title,
                        page_number, l_start, l_end,
                    )
                    current_path = path
                    elements.append(
                        DocumentElement(
                            kind="heading",
                            text=title,
                            span=SourceSpan(
                                page_number=page_number,
                                char_start=l_start,
                                char_end=l_end,
                            ),
                            order=order,
                            level=level,
                            section_path=path,
                        )
                    )
                    order += 1
                    i += 1
                    continue

                if _is_table_line(line):
                    table_lines = []
                    table_start = l_start
                    table_end = l_end
                    j = i
                    while j < len(lines) and _is_table_line(lines[j][0]):
                        table_lines.append(lines[j])
                        table_end = lines[j][2]
                        j += 1
                    elements.append(
                        self._build_table_element(
                            table_lines, page_number, order, current_path
                        )
                    )
                    order += 1
                    i = j
                    continue

                # Paragraph: accumulate consecutive non-empty, non-special lines.
                para_lines = []
                para_start = l_start
                para_end = l_end
                j = i
                while j < len(lines):
                    nxt = lines[j][0]
                    if not nxt.strip():
                        break
                    if _heading_level_and_title(nxt):
                        break
                    if _LIST_ITEM_RE.match(nxt):
                        break
                    if _is_table_line(nxt):
                        break
                    para_lines.append(nxt.strip())
                    para_end = lines[j][2]
                    j += 1
                elements.append(
                    DocumentElement(
                        kind="paragraph",
                        text=" ".join(para_lines),
                        span=SourceSpan(
                            page_number=page_number,
                            char_start=para_start,
                            char_end=para_end,
                        ),
                        order=order,
                        section_path=current_path,
                    )
                )
                order += 1
                i = j

        return StructuredDocument(
            document_id=document_id,
            filename=filename,
            page_count=len(pages),
            elements=elements,
            sections=top_sections,
            extraction_warnings=warnings,
        )

    @staticmethod
    def _in_list_context(elements: List[DocumentElement]) -> bool:
        """True when the previous element opens or continues a list."""
        if not elements:
            return False
        prev = elements[-1]
        if prev.kind == "list_item":
            return True
        if prev.kind == "paragraph" and prev.text.rstrip().endswith(":"):
            return True
        return False

    @staticmethod
    def _push_section(
        stack: List[Section],
        top: List[Section],
        level: int,
        title: str,
        page_number: int,
        start: int,
        end: int,
    ) -> str:
        """Push a section onto the hierarchy stack and return its dotted path."""
        while stack and stack[-1].level >= level:
            stack.pop()
        parent_path = stack[-1].path if stack else ""
        path = f"{parent_path}.{title}" if parent_path else title
        section = Section(
            title=title,
            level=level,
            path=path,
            page_number=page_number,
            span=SourceSpan(
                page_number=page_number, char_start=start, char_end=end
            ),
        )
        if stack:
            stack[-1].children.append(section)
        else:
            top.append(section)
        stack.append(section)
        return path

    @staticmethod
    def _build_table_element(
        table_lines,
        page_number: int,
        order: int,
        section_path: Optional[str],
    ) -> DocumentElement:
        rows: List[List[TableCell]] = []
        for r_idx, (line, l_start, l_end) in enumerate(table_lines):
            cells: List[TableCell] = []
            for c_idx, cell_text in enumerate(_split_table_cells(line)):
                cells.append(
                    TableCell(
                        text=cell_text,
                        row=r_idx,
                        col=c_idx,
                        span=SourceSpan(
                            page_number=page_number,
                            char_start=l_start,
                            char_end=l_end,
                        ),
                    )
                )
            rows.append(cells)
        table = StructuredTable(
            rows=rows,
            span=SourceSpan(
                page_number=page_number,
                char_start=table_lines[0][1],
                char_end=table_lines[-1][2],
            ),
        )
        return DocumentElement(
            kind="table",
            text="\n".join(line for line, _, _ in table_lines),
            span=table.span,
            order=order,
            section_path=section_path,
            table=table,
        )


def reconstruct_pages_from_chunks(chunks: List[dict]) -> List[dict]:
    """Rebuild per-page segments from stored chunk source references.

    Chunks carry absolute ``character_start``/``character_end`` offsets into
    the normalized document text plus a ``page_number``.  Placing each chunk's
    text at its absolute offset reconstructs the normalized text exactly
    (overlapping chunks write identical characters).  Page boundaries are
    derived from the min/max offsets observed per page.

    This reuses the source references captured at ingestion time; the original
    file is not re-read.

    Reliability guarantees (enforced, not silent):

    - A chunk whose ``text`` length does not match
      ``character_end - character_start`` is rejected (inconsistent chunk).
    - Overlapping chunks must agree on every shared character; otherwise the
      reconstruction is rejected (conflicting source text).
    - Every character position inside a derived page range must be covered by
      at least one chunk; otherwise the reconstruction is rejected (gap in
      source coverage).

    Args:
        chunks: Chunk dicts as returned by the vector store
            (``character_start``, ``character_end``, ``page_number``, ``text``).

    Returns:
        A list of page dicts (``page_number``, ``text``, ``character_start``,
        ``character_end``) suitable for :meth:`StructuredTextParser.parse`.

    Raises:
        ValueError: If chunks are missing/invalid, overlap conflicts, or
            leave a gap in source coverage, so unreliable citations cannot be
            produced silently.
    """
    usable = [
        c
        for c in chunks
        if c.get("character_start") is not None
        and c.get("character_end") is not None
    ]
    if not usable:
        raise ValueError("No usable source references found in chunks")

    # Validate each chunk's span/text agreement.
    for c in usable:
        start = int(c["character_start"])
        end = int(c["character_end"])
        text = c.get("text", "") or ""
        if end < start:
            raise ValueError(
                f"Chunk has invalid span: character_end ({end}) < "
                f"character_start ({start})"
            )
        if (end - start) != len(text):
            raise ValueError(
                f"Chunk span/text mismatch: span length {end - start} "
                f"but text length {len(text)}"
            )

    chars: dict[int, str] = {}
    # Track which chunk wrote each position to detect overlaps that disagree.
    owners: dict[int, str] = {}
    for c in usable:
        start = int(c["character_start"])
        text = c.get("text", "") or ""
        chunk_id = str(c.get("chunk_id") or c.get("id") or id(c))
        for i, ch in enumerate(text):
            pos = start + i
            if pos in chars and chars[pos] != ch:
                raise ValueError(
                    f"Conflicting source text at offset {pos}: "
                    f"{chars[pos]!r} vs {ch!r}"
                )
            chars[pos] = ch
            owners.setdefault(pos, chunk_id)

    if not chars:
        raise ValueError("No usable source references found in chunks")

    max_end = max(chars) + 1
    full_text = "".join(chars.get(i, "\n") for i in range(max_end))

    page_bounds: dict[int, dict] = {}
    for c in usable:
        pn = c.get("page_number") or 1
        start = int(c["character_start"])
        end = int(c["character_end"])
        entry = page_bounds.get(pn)
        if entry is None:
            page_bounds[pn] = {
                "page_number": pn,
                "character_start": start,
                "character_end": end,
            }
        else:
            entry["character_start"] = min(entry["character_start"], start)
            entry["character_end"] = max(entry["character_end"], end)

    pages: List[dict] = []
    for pn in sorted(page_bounds):
        entry = page_bounds[pn]
        # Gap check: every position in the page range must be covered.
        for pos in range(entry["character_start"], entry["character_end"]):
            if pos not in chars:
                raise ValueError(
                    f"Gap in source coverage on page {pn} at offset {pos}"
                )
        entry["text"] = full_text[
            entry["character_start"]:entry["character_end"]
        ]
        pages.append(entry)
    return pages