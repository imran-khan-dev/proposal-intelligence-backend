"""Pytest configuration for Proposal Intelligence tests."""

import os
import sys

# Set test environment variables before any imports
os.environ.setdefault("AI_PROVIDER", "ollama")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("OLLAMA_MODEL", "test-model")
os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
os.environ.setdefault("VECTOR_STORE_TYPE", "memory")

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))