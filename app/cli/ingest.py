"""CLI command to ingest company knowledge documents into the vector store."""

import argparse
import sys
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from app.ai.embedding_factory import get_embedding_provider
from app.rag.ingest import ingest_document
from app.rag.store_factory import get_vector_store
from app.core.database import initialize_database, is_database_initialized


SUPPORTED_EXTENSIONS = {".txt", ".md"}


def derive_doc_id(file_path: Path) -> str:
    """Derive a stable document ID from the file path.

    Uses the relative path from the current working directory,
    with extension removed and separators replaced with underscores.
    """
    try:
        rel_path = file_path.relative_to(Path.cwd())
    except ValueError:
        # File is outside cwd, use just the stem
        return file_path.stem

    # Remove extension and replace separators
    doc_id = str(rel_path.with_suffix("")).replace(os.sep, "_")
    return doc_id


def read_text_file(file_path: Path) -> str:
    """Read a text file with explicit UTF-8 encoding.

    Falls back to latin-1 if UTF-8 fails, which can decode any byte sequence.
    """
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback for files with mixed encodings
        return file_path.read_text(encoding="latin-1")


def main(argv: list[str] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli.ingest",
        description="Ingest a company knowledge document into the configured vector store.",
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to the document file (.txt or .md) to ingest.",
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Optional document ID. Defaults to a stable ID derived from the file path.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Maximum chunk size in characters (default: 800).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Overlap between chunks in characters (default: 100).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output.",
    )

    args = parser.parse_args(argv)

    # Validate file
    if not args.file.exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        return 1

    if not args.file.is_file():
        print(f"Error: Not a file: {args.file}", file=sys.stderr)
        return 1

    suffix = args.file.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        print(
            f"Error: Unsupported file extension: {suffix}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
            file=sys.stderr,
        )
        return 1

    # Derive document ID
    doc_id = args.doc_id or derive_doc_id(args.file)

    # Read file content
    try:
        text = read_text_file(args.file)
    except Exception as exc:
        print(f"Error reading file: {exc}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Ingesting: {args.file}")
        print(f"Document ID: {doc_id}")
        print(f"File size: {args.file.stat().st_size} bytes")
        print(f"Chunk size: {args.chunk_size}, Overlap: {args.overlap}")

    # Initialize database if using PostgreSQL vector store
    store_type = os.environ.get("VECTOR_STORE_TYPE", "memory").lower()
    if store_type == "postgresql" and not is_database_initialized():
        initialize_database()

    # Initialize embedding provider and vector store via factories
    try:
        embedder = get_embedding_provider()
        store = get_vector_store()

        if args.verbose:
            print(f"Embedding provider: {type(embedder).__name__} (dim={embedder.dimension})")
            print(f"Vector store: {type(store).__name__}")

        # Ingest the document
        chunk_count = ingest_document(
            text=text,
            doc_id=doc_id,
            store=store,
            embedder=embedder,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )

        if args.verbose:
            print(f"Chunks ingested: {chunk_count}")

        print(f"Successfully ingested '{doc_id}' ({chunk_count} chunks)")
        return 0

    except RuntimeError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())