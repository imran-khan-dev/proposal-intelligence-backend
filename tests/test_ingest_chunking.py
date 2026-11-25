"""Unit tests for page-aware chunking and source-reference offsets.

These tests are pure (no database, no network, no API keys) and exercise the
offset convention: character offsets refer to the original normalized document
text, never to chunk-relative positions.
"""

import pytest

from app.rag.ingest import (
    TextChunk,
    build_normalized_text,
    chunk_pages,
    chunk_text_with_offsets,
    extract_pdf_pages,
)


class TestChunkTextWithOffsets:
    def test_single_chunk_when_text_fits(self):
        chunks = chunk_text_with_offsets("hello world", chunk_size=800, overlap=100)
        assert len(chunks) == 1
        assert chunks[0].text == "hello world"
        assert chunks[0].character_start == 0
        assert chunks[0].character_end == len("hello world")
        assert chunks[0].page_number is None

    def test_offsets_are_absolute_with_base_offset(self):
        chunks = chunk_text_with_offsets(
            "abcdefghij", chunk_size=4, overlap=1, base_offset=100
        )
        # chunk 0: [0,4) -> abs [100,104)
        # chunk 1: [3,7) -> abs [103,107)
        # chunk 2: [6,10) -> abs [106,110)
        assert len(chunks) == 3
        assert chunks[0].character_start == 100
        assert chunks[0].character_end == 104
        assert chunks[1].character_start == 103
        assert chunks[1].character_end == 107
        assert chunks[2].character_start == 106
        assert chunks[2].character_end == 110
        # Offsets refer to normalized text, not chunk-relative positions.
        assert chunks[1].character_start != 0

    def test_page_number_preserved(self):
        chunks = chunk_text_with_offsets(
            "some text", chunk_size=800, overlap=100, page_number=3
        )
        assert len(chunks) == 1
        assert chunks[0].page_number == 3

    def test_section_path_preserved(self):
        chunks = chunk_text_with_offsets(
            "some text",
            chunk_size=800,
            overlap=100,
            section_path="3.2.1",
        )
        assert chunks[0].section_path == "3.2.1"

    def test_empty_text_returns_no_chunks(self):
        assert chunk_text_with_offsets("", chunk_size=800, overlap=100) == []
        assert chunk_text_with_offsets("   ", chunk_size=800, overlap=100) == []

    def test_invalid_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_text_with_offsets("a", chunk_size=0, overlap=0)
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            chunk_text_with_offsets("a", chunk_size=4, overlap=-1)
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            chunk_text_with_offsets("a", chunk_size=4, overlap=4)


class TestChunkPages:
    def test_chunks_never_cross_page_boundary(self):
        pages = [
            {"page_number": 1, "text": "alpha beta gamma delta epsilon"},
            {"page_number": 2, "text": "zeta eta theta iota kappa"},
        ]
        chunks = chunk_pages(pages, chunk_size=20, overlap=2)
        page_numbers = {c.page_number for c in chunks}
        assert page_numbers == {1, 2}
        # Every chunk from page 1 must lie within page 1's offset range.
        normalized = build_normalized_text(pages)
        page1_end = len(pages[0]["text"])  # before the "\n\n" separator
        for c in chunks:
            if c.page_number == 1:
                assert c.character_end <= page1_end
            else:
                assert c.character_start >= page1_end + 2

    def test_offsets_match_normalized_text(self):
        pages = [
            {"page_number": 1, "text": "hello world"},
            {"page_number": 2, "text": "foo bar"},
        ]
        normalized = build_normalized_text(pages)
        chunks = chunk_pages(pages, chunk_size=100, overlap=0)
        for c in chunks:
            # The slice of the normalized text at the chunk's offsets must equal
            # the chunk text.
            assert normalized[c.character_start:c.character_end] == c.text

    def test_page_number_defaults_to_index(self):
        pages = [{"text": "only page"}]
        chunks = chunk_pages(pages, chunk_size=100, overlap=0)
        assert chunks[0].page_number == 1

    def test_empty_pages_produce_no_chunks(self):
        pages = [{"page_number": 1, "text": ""}, {"page_number": 2, "text": "real text"}]
        chunks = chunk_pages(pages, chunk_size=100, overlap=0)
        assert len(chunks) == 1
        assert chunks[0].page_number == 2

    def test_base_metadata_merged(self):
        pages = [{"page_number": 1, "text": "hello world"}]
        chunks = chunk_pages(
            pages,
            chunk_size=100,
            overlap=0,
            base_metadata={"source_filename": "a.pdf", "doc_id": "doc1"},
        )
        assert chunks[0].metadata == {"source_filename": "a.pdf", "doc_id": "doc1"}

    def test_explicit_character_offsets_honored(self):
        pages = [
            {
                "page_number": 1,
                "text": "page one text",
                "character_start": 500,
                "character_end": 513,
            }
        ]
        chunks = chunk_pages(pages, chunk_size=100, overlap=0)
        assert chunks[0].character_start == 500
        assert chunks[0].character_end == 513


class TestBuildNormalizedText:
    def test_pages_joined_with_double_newline(self):
        pages = [{"text": "ab"}, {"text": "cd"}]
        assert build_normalized_text(pages) == "ab\n\ncd"

    def test_empty_pages_list(self):
        assert build_normalized_text([]) == ""


class TestExtractPdfPages:
    def test_extracts_pages_with_offsets(self, tmp_path):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from fixtures.pdfgen import build_pdf

        pdf_path = tmp_path / "sample.pdf"
        pdf_path.write_bytes(
            build_pdf(
                [
                    ["page one line one", "page one line two"],
                    ["page two line one"],
                ]
            )
        )

        pages = extract_pdf_pages(str(pdf_path))
        assert len(pages) == 2
        assert pages[0]["page_number"] == 1
        assert "page one line one" in pages[0]["text"]
        assert pages[0]["character_start"] == 0
        assert pages[0]["character_end"] == len(pages[0]["text"])
        assert pages[1]["page_number"] == 2
        assert pages[1]["character_start"] > pages[0]["character_end"]

    def test_rejects_pdf_with_no_pages(self, tmp_path):
        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")
        with pytest.raises(ValueError, match="Unable to read PDF"):
            extract_pdf_pages(str(pdf_path))


class TestTextChunk:
    def test_dataclass_defaults(self):
        chunk = TextChunk(text="hi")
        assert chunk.page_number is None
        assert chunk.section_path is None
        assert chunk.character_start == 0
        assert chunk.character_end == 0
        assert chunk.metadata is None
