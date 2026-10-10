"""Tests for text chunking utilities."""

import pytest

from app.rag.ingest import chunk_text


class TestChunkText:
    def test_empty_input_returns_empty_list(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   ") == []
        assert chunk_text("\n\t") == []

    def test_text_shorter_than_chunk_size_returns_one_chunk(self):
        text = "Short text."
        result = chunk_text(text, chunk_size=100, overlap=20)
        assert result == [text]

    def test_text_equal_to_chunk_size_returns_one_chunk(self):
        text = "x" * 100
        result = chunk_text(text, chunk_size=100, overlap=20)
        assert result == [text]
        assert len(result[0]) == 100

    def test_long_text_produces_multiple_chunks_within_size_limit(self):
        text = "a" * 2000
        chunks = chunk_text(text, chunk_size=800, overlap=100)
        assert len(chunks) == 3
        for chunk in chunks:
            assert len(chunk) <= 800

    def test_consecutive_chunks_overlap(self):
        text = "0123456789" * 100  # 1000 chars
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        assert len(chunks) == 4
        # Each consecutive pair should overlap by 50 chars
        for i in range(len(chunks) - 1):
            # The end of chunk i should match the start of chunk i+1 for 50 chars
            assert chunks[i][-50:] == chunks[i + 1][:50]

    def test_reconstruction_preserves_original_text(self):
        """Original text can be reconstructed by stitching chunks with overlap removed."""
        text = "This is a test document. " * 50  # ~1200 chars
        chunks = chunk_text(text, chunk_size=500, overlap=100)

        # Reconstruct by taking first chunk fully, then appending only the non-overlapping
        # suffix of each subsequent chunk
        reconstructed = chunks[0]
        for chunk in chunks[1:]:
            reconstructed += chunk[100:]  # skip overlap

        assert reconstructed == text

    def test_reconstruction_with_different_overlap(self):
        text = "abcdefghijklmnopqrstuvwxyz" * 20  # 520 chars
        chunks = chunk_text(text, chunk_size=150, overlap=30)

        reconstructed = chunks[0]
        for chunk in chunks[1:]:
            reconstructed += chunk[30:]

        assert reconstructed == text

    def test_invalid_chunk_size_zero_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_text("text", chunk_size=0, overlap=10)

    def test_invalid_chunk_size_negative_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_text("text", chunk_size=-1, overlap=10)

    def test_invalid_overlap_negative_raises(self):
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            chunk_text("text", chunk_size=100, overlap=-1)

    def test_invalid_overlap_equal_to_chunk_size_raises(self):
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            chunk_text("text", chunk_size=100, overlap=100)

    def test_invalid_overlap_greater_than_chunk_size_raises(self):
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            chunk_text("text", chunk_size=100, overlap=150)

    def test_no_infinite_loop_with_valid_params(self):
        # Large text, small chunk size, large overlap (but still < chunk_size)
        text = "x" * 10000
        chunks = chunk_text(text, chunk_size=200, overlap=199)
        # Should make progress: each step advances by (chunk_size - overlap) = 1 char
        assert len(chunks) == 10000 - 199  # ~9801 chunks
        # Verify we don't hang - just check it completes quickly
        assert all(len(c) <= 200 for c in chunks)

    def test_single_character_overlap(self):
        text = "abcdefg"
        chunks = chunk_text(text, chunk_size=4, overlap=1)
        # "abcd" + "defg" with 1 char overlap
        assert chunks == ["abcd", "defg"]

    def test_chunk_size_one_with_zero_overlap(self):
        text = "abc"
        chunks = chunk_text(text, chunk_size=1, overlap=0)
        assert chunks == ["a", "b", "c"]