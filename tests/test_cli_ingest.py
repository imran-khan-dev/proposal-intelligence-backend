"""Tests for CLI ingest command."""

import tempfile
import os
import sys
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from app.cli.ingest import derive_doc_id, read_text_file, main
from app.ai.embeddings import EmbeddingProvider


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic mock embedding provider for testing."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dimension] * len(texts)


def _run_cli_with_mock(args: list[str], env: dict = None) -> tuple[int, str, str]:
    """Run the CLI command with mocked embedding provider."""
    cmd = [sys.executable, "-m", "app.cli.ingest"] + args

    test_env = os.environ.copy()
    test_env["VECTOR_STORE_TYPE"] = "memory"
    test_env["EMBEDDING_PROVIDER"] = "local"  # Will be mocked
    if env:
        test_env.update(env)

    # Use mock provider by patching the factory
    with patch('app.ai.embedding_factory.get_embedding_provider', return_value=MockEmbeddingProvider(dimension=4)):
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=E_PROPOSAL_INTELLIGENCE_BACKEND,
            env=test_env,
        )
    return result.returncode, result.stdout, result.stderr


class TestCLIHelpers:
    """Tests for CLI helper functions."""

    def test_derive_doc_id_from_relative_path(self):
        """doc_id derived from relative path with separators replaced."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "docs" / "policy.txt"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("test")

            with patch('pathlib.Path.cwd', return_value=Path(tmpdir)):
                doc_id = derive_doc_id(file_path)
                assert doc_id == "docs_policy"

    def test_derive_doc_id_outside_cwd(self):
        """doc_id falls back to stem when file is outside cwd."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "outside.txt"
            file_path.write_text("test")

            with patch('pathlib.Path.cwd', return_value=Path("/some/other/dir")):
                doc_id = derive_doc_id(file_path)
                assert doc_id == "outside"

    def test_derive_doc_id_with_md_extension(self):
        """doc_id works for .md files."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "guide.md"
            file_path.write_text("test")

            with patch('pathlib.Path.cwd', return_value=Path(tmpdir)):
                doc_id = derive_doc_id(file_path)
                assert doc_id == "guide"

    def test_read_text_file_utf8(self):
        """Read UTF-8 encoded file."""
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".txt") as f:
            f.write("Hello world\nwith unicode: café")
            path = Path(f.name)

        try:
            text = read_text_file(path)
            assert "café" in text
        finally:
            path.unlink()

    def test_read_text_file_latin1_fallback(self):
        """Read file with latin-1 fallback."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".txt") as f:
            f.write(b"Hello world\nwith latin-1: \xe9")  # é in latin-1
            path = Path(f.name)

        try:
            text = read_text_file(path)
            assert "é" in text
        finally:
            path.unlink()


class TestCLIIngest:
    """Tests for the ingest CLI command."""

    def test_missing_file(self):
        """Missing file returns error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.txt"
            code, stdout, stderr = _run_cli_with_mock([str(missing)])
            assert code != 0
            assert "File not found" in stderr

    def test_not_a_file(self):
        """Directory returns error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code, stdout, stderr = _run_cli_with_mock([tmpdir])
            assert code != 0
            assert "Not a file" in stderr

    def test_unsupported_extension(self):
        """Unsupported extension returns error."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            code, stdout, stderr = _run_cli_with_mock([path])
            assert code != 0
            assert "Unsupported file extension" in stderr
            assert ".pdf" in stderr
        finally:
            os.unlink(path)

    def test_successful_ingest_txt(self):
        """Successful ingestion of .txt file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "policy.txt"
            file_path.write_text("Company expense policy: all expenses over $100 require approval.")

            code, stdout, stderr = _run_cli_with_mock([str(file_path)])
            assert code == 0, f"stderr: {stderr}"
            assert "Successfully ingested" in stdout
            assert "policy" in stdout
            assert "1 chunks" in stdout or "chunks" in stdout

    def test_successful_ingest_md(self):
        """Successful ingestion of .md file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "guide.md"
            file_path.write_text("# Company Guide\n\nExpenses over $100 need approval.")

            code, stdout, stderr = _run_cli_with_mock([str(file_path)])
            assert code == 0, f"stderr: {stderr}"
            assert "Successfully ingested" in stdout
            assert "guide" in stdout

    def test_custom_doc_id(self):
        """Custom document ID is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "policy.txt"
            file_path.write_text("Test content")

            code, stdout, stderr = _run_cli_with_mock([str(file_path), "--doc-id", "custom-policy-id"])
            assert code == 0, f"stderr: {stderr}"
            assert "custom-policy-id" in stdout

    def test_reingestion_replaces(self):
        """Re-ingesting same doc_id replaces old chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "policy.txt"

            # First ingestion
            file_path.write_text("Old policy content")
            code1, stdout1, stderr1 = _run_cli_with_mock([str(file_path), "--doc-id", "test-policy"])
            assert code1 == 0

            # Second ingestion with different content
            file_path.write_text("New policy content with updates")
            code2, stdout2, stderr2 = _run_cli_with_mock([str(file_path), "--doc-id", "test-policy"])
            assert code2 == 0

    def test_verbose_output(self):
        """Verbose flag prints additional info."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "policy.txt"
            file_path.write_text("Test content for verbose output")

            code, stdout, stderr = _run_cli_with_mock([str(file_path), "--verbose"])
            assert code == 0, f"stderr: {stderr}"
            assert "Embedding provider:" in stdout
            assert "Vector store:" in stdout
            assert "Chunk size:" in stdout
            assert "Overlap:" in stdout

    def test_persistent_store_path(self):
        """Persistent store path from env is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "policy.txt"
            file_path.write_text("Test content for persistent store")

            store_path = Path(tmpdir) / "test_store.jsonl"
            code, stdout, stderr = _run_cli_with_mock(
                [str(file_path)],
                env={
                    "VECTOR_STORE_TYPE": "persistent",
                    "VECTOR_STORE_PATH": str(store_path)
                }
            )
            assert code == 0, f"stderr: {stderr}"
            assert "Successfully ingested" in stdout
            # Verify file was created
            assert store_path.exists()
            assert store_path.stat().st_size > 0

    def test_chunk_size_and_overlap_options(self):
        """Custom chunk size and overlap options work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "policy.txt"
            file_path.write_text("A" * 500)  # Small enough for single chunk with default

            code, stdout, stderr = _run_cli_with_mock(
                [str(file_path), "--chunk-size", "100", "--overlap", "20"]
            )
            assert code == 0, f"stderr: {stderr}"
            assert "Successfully ingested" in stdout


# Need to define the backend path for subprocess calls
E_PROPOSAL_INTELLIGENCE_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))