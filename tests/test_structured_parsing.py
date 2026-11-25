"""Unit tests for structured RFP parsing.

Covers reading order, section hierarchy, lists, tables, source offsets, empty
and poorly extracted pages, and reconstruction of pages from stored chunks.
No PDF, network, or model dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import rfp_fixtures as fx
from app.modules.requirements.parsing import (
    StructuredTextParser,
    reconstruct_pages_from_chunks,
)
from app.modules.requirements.structured_document import StructuredDocument


def _parse(pages, doc_id="doc1", filename="rfp.pdf"):
    return StructuredTextParser().parse(doc_id, filename, pages)


class TestReadingOrderAndStructure:
    def test_elements_in_reading_order(self):
        doc = _parse(fx.numbered_fixture())
        kinds = [e.kind for e in doc.elements]
        # First element is the "1. Scope of Work" heading.
        assert kinds[0] == "heading"
        assert doc.elements[0].text == "Scope of Work"
        # Orders are strictly increasing.
        orders = [e.order for e in doc.elements]
        assert orders == sorted(orders)
        assert orders == list(range(len(orders)))

    def test_section_hierarchy_nested(self):
        doc = _parse(fx.nested_fixture())
        assert doc.sections, "expected at least one top-level section"
        top = doc.sections[0]
        assert top.title == "Technical Requirements"
        assert top.level == 1
        # 3.1 Infrastructure is a child of 3.
        child_titles = [c.title for c in top.children]
        assert "Infrastructure" in child_titles
        infra = next(c for c in top.children if c.title == "Infrastructure")
        # 3.1.1 (a list intro) is a child of 3.1.
        assert infra.children, "expected nested child under 3.1"

    def test_numbered_requirement_not_treated_as_heading(self):
        doc = _parse(fx.numbered_fixture())
        headings = [e.text for e in doc.elements if e.kind == "heading"]
        # "The system must support SAML 2.0 single sign-on." must not be a heading.
        assert all("must support" not in h for h in headings)
        assert "Functional Requirements" in headings

    def test_list_items_detected(self):
        doc = _parse(fx.nested_fixture())
        list_items = [e for e in doc.elements if e.kind == "list_item"]
        texts = [e.text for e in list_items]
        assert "Encryption of data at rest and in transit" in texts
        assert "Role-based access control" in texts
        markers = {e.list_marker for e in list_items}
        assert "-" in markers
        assert "1." in markers

    def test_table_rows_and_cells_preserved(self):
        doc = _parse(fx.table_fixture())
        tables = [e for e in doc.elements if e.kind == "table"]
        assert len(tables) == 1
        table = tables[0].table
        assert table is not None
        assert len(table.rows) == 4  # header + 3 requirement rows
        header = [c.text for c in table.rows[0]]
        assert header == ["ID", "Requirement", "Priority"]
        first_req = [c.text for c in table.rows[1]]
        assert first_req[0] == "REQ-001"
        assert "SAML 2.0" in first_req[1]

    def test_paragraph_text_and_offsets(self):
        pages = fx.make_pages(["The contractor shall provide a portal."])
        doc = _parse(pages)
        paras = [e for e in doc.elements if e.kind == "paragraph"]
        assert len(paras) == 1
        p = paras[0]
        assert p.span.char_start == 0
        assert p.span.char_end == len(pages[0]["text"])
        # Offsets map back to the source text exactly.
        assert pages[0]["text"][p.span.char_start:p.span.char_end] == p.text


class TestEmptyAndPoorPages:
    def test_empty_page_recorded_as_warning(self):
        doc = _parse(fx.empty_pages_fixture())
        warnings = " ".join(doc.extraction_warnings)
        assert "no extractable text" in warnings
        assert "very little extractable text" in warnings

    def test_empty_page_does_not_break_reading_order(self):
        doc = _parse(fx.empty_pages_fixture())
        # Elements from the non-empty page are still present and ordered.
        assert any(e.kind == "heading" for e in doc.elements)
        orders = [e.order for e in doc.elements]
        assert orders == sorted(orders)


class TestChunkReconstruction:
    def test_reconstruct_pages_from_chunks_roundtrip(self):
        pages = fx.multi_page_fixture()
        # Simulate stored chunks: one chunk per page covering the whole page.
        chunks = []
        for p in pages:
            chunks.append(
                {
                    "doc_id": "doc1",
                    "chunk_index": p["page_number"] - 1,
                    "page_number": p["page_number"],
                    "character_start": p["character_start"],
                    "character_end": p["character_end"],
                    "text": p["text"],
                    "source_filename": "rfp.pdf",
                }
            )
        rebuilt = reconstruct_pages_from_chunks(chunks)
        assert len(rebuilt) == len(pages)
        for original, restored in zip(pages, rebuilt):
            assert restored["page_number"] == original["page_number"]
            assert restored["text"] == original["text"]
            assert restored["character_start"] == original["character_start"]

    def test_reconstruct_handles_overlapping_chunks(self):
        pages = fx.make_pages(["alpha beta gamma delta"])
        text = pages[0]["text"]
        base = pages[0]["character_start"]
        # Two overlapping chunks covering the page.
        chunks = [
            {
                "page_number": 1,
                "character_start": base,
                "character_end": base + 10,
                "text": text[:10],
            },
            {
                "page_number": 1,
                "character_start": base + 5,
                "character_end": base + len(text),
                "text": text[5:],
            },
        ]
        rebuilt = reconstruct_pages_from_chunks(chunks)
        assert rebuilt[0]["text"] == text

    def test_reconstruct_empty_chunks_raises(self):
        import pytest

        with pytest.raises(ValueError, match="No usable source references"):
            reconstruct_pages_from_chunks([])

    def test_reconstruct_gap_raises(self):
        import pytest

        pages = fx.make_pages(["alpha beta gamma delta"])
        text = pages[0]["text"]
        base = pages[0]["character_start"]
        # Two chunks with a hole between them -> internal gap.
        chunks = [
            {
                "page_number": 1,
                "character_start": base,
                "character_end": base + 5,
                "text": text[:5],
            },
            {
                "page_number": 1,
                "character_start": base + 10,
                "character_end": base + len(text),
                "text": text[10:],
            },
        ]
        with pytest.raises(ValueError, match="Gap in source coverage"):
            reconstruct_pages_from_chunks(chunks)

    def test_reconstruct_conflict_raises(self):
        import pytest

        pages = fx.make_pages(["alpha beta gamma delta"])
        text = pages[0]["text"]
        base = pages[0]["character_start"]
        chunks = [
            {
                "page_number": 1,
                "character_start": base,
                "character_end": base + len(text),
                "text": text,
            },
            {
                "page_number": 1,
                "character_start": base,
                "character_end": base + 5,
                "text": "XXXXX",
            },
        ]
        with pytest.raises(ValueError, match="Conflicting source text"):
            reconstruct_pages_from_chunks(chunks)

    def test_reconstruct_span_text_mismatch_raises(self):
        import pytest

        pages = fx.make_pages(["alpha beta gamma delta"])
        base = pages[0]["character_start"]
        chunks = [
            {
                "page_number": 1,
                "character_start": base,
                "character_end": base + 10,
                "text": "short",
            },
        ]
        with pytest.raises(ValueError, match="span/text mismatch"):
            reconstruct_pages_from_chunks(chunks)

    def test_reconstruct_stable_across_repeated_calls(self):
        pages = fx.multi_page_fixture()
        chunks = []
        for p in pages:
            chunks.append(
                {
                    "doc_id": "doc1",
                    "chunk_index": p["page_number"] - 1,
                    "page_number": p["page_number"],
                    "character_start": p["character_start"],
                    "character_end": p["character_end"],
                    "text": p["text"],
                    "source_filename": "rfp.pdf",
                }
            )
        first = reconstruct_pages_from_chunks(chunks)
        second = reconstruct_pages_from_chunks(chunks)
        assert first == second


class TestStructuredDocumentModel:
    def test_document_fields(self):
        doc = _parse(fx.numbered_fixture(), doc_id="abc", filename="x.pdf")
        assert isinstance(doc, StructuredDocument)
        assert doc.document_id == "abc"
        assert doc.filename == "x.pdf"
        assert doc.page_count == 1