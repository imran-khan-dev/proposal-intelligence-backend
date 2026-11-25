"""Structured document representation for RFP parsing.

This module defines the in-memory representation of a parsed RFP document.
It preserves the information the downstream requirement extractor needs:

- document id and filename
- page numbers
- section hierarchy and titles
- paragraphs
- numbered and bulleted list items
- tables (rows and cells) where extraction permits
- source text and absolute character offsets
- reading order

The representation is intentionally decoupled from any specific parser.  A
parser (see :mod:`app.modules.requirements.parsing`) produces a
:class:`StructuredDocument` from per-page text segments.  Nothing here
re-implements ingestion, chunking, or embedding.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class SourceSpan(BaseModel):
    """A character span within the normalized document text.

    ``char_start``/``char_end`` are absolute offsets into the normalized
    document text (the same text the ingestion pipeline chunks), so a span can
    be mapped back to the exact source location of a requirement.
    """

    page_number: int = Field(..., description="1-based page number")
    char_start: int = Field(..., description="Absolute start offset (inclusive)")
    char_end: int = Field(..., description="Absolute end offset (exclusive)")


class TableCell(BaseModel):
    """A single table cell with its text and (best-effort) source span."""

    text: str = Field(..., description="Cell text")
    row: int = Field(..., description="0-based row index")
    col: int = Field(..., description="0-based column index")
    span: Optional[SourceSpan] = Field(
        None, description="Best-effort source span for the cell text"
    )


class StructuredTable(BaseModel):
    """A table extracted from the document, preserving rows and cells."""

    rows: List[List[TableCell]] = Field(
        default_factory=list, description="Rows of cells, in reading order"
    )
    span: SourceSpan = Field(..., description="Source span covering the table")


class DocumentElement(BaseModel):
    """One element of the document in reading order.

    ``kind`` is one of ``heading``, ``paragraph``, ``list_item``, ``table``.
    """

    kind: str = Field(..., description="Element kind")
    text: str = Field(..., description="Element text")
    span: SourceSpan = Field(..., description="Source span of the element")
    order: int = Field(..., description="Reading-order index (0-based)")
    level: Optional[int] = Field(
        None, description="Heading level (1-based), for headings"
    )
    section_path: Optional[str] = Field(
        None, description="Dotted section path the element belongs to"
    )
    list_marker: Optional[str] = Field(
        None, description="List marker (e.g. '1.', '-', 'a)'), for list items"
    )
    table: Optional[StructuredTable] = Field(
        None, description="Structured table payload, for table elements"
    )


class Section(BaseModel):
    """A section in the document hierarchy."""

    title: str = Field(..., description="Section title")
    level: int = Field(..., description="Section level (1-based)")
    path: str = Field(..., description="Dotted section path (e.g. '1.2.1')")
    page_number: int = Field(..., description="1-based page number")
    span: SourceSpan = Field(..., description="Source span of the heading")
    children: List["Section"] = Field(
        default_factory=list, description="Nested child sections"
    )


class StructuredDocument(BaseModel):
    """A parsed, structured representation of an RFP document."""

    document_id: str = Field(..., description="Stable document identifier")
    filename: str = Field(..., description="Original source filename")
    page_count: int = Field(..., description="Number of pages parsed")
    elements: List[DocumentElement] = Field(
        default_factory=list,
        description="Document elements in reading order",
    )
    sections: List[Section] = Field(
        default_factory=list, description="Top-level section hierarchy"
    )
    extraction_warnings: List[str] = Field(
        default_factory=list,
        description="Warnings about empty/poorly extracted content",
    )


Section.model_rebuild()