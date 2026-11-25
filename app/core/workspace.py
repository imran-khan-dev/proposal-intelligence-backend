"""Centralized workspace resolution for the single-tenant MVP.

Workspace identity is resolved from server configuration, never from
client-supplied input.  This prevents a client from selecting another
workspace by changing a query parameter or header.

Resolution order:
  1. ``WORKSPACE_ID`` environment variable
  2. ``TENANT_ID`` environment variable (legacy alias)
  3. ``"default-workspace"`` (single-tenant default)

This module is the single place where workspace identity is derived.  All
document operations (ingestion, listing, retrieval, chunk lookup, deletion,
re-ingestion) must call :func:`resolve_workspace_id` rather than trusting a
client-provided value.  Keeping the resolver here (rather than inside a
particular store or controller) preserves a straightforward migration path to
authenticated multi-tenant access later: the resolver can be replaced with a
dependency that reads the workspace from a validated session/token without
touching the document layer.
"""

from __future__ import annotations

import os


def resolve_workspace_id() -> str:
    """Return the server-resolved workspace identifier for this process.

    Reads environment configuration only.  Client input is never trusted,
    so a caller cannot access another workspace by supplying a different
    ``workspace_id``.

    Returns:
        A non-empty workspace identifier string.
    """
    return (
        os.environ.get("WORKSPACE_ID")
        or os.environ.get("TENANT_ID")
        or "default-workspace"
    )
