"""Centralized workspace resolution for the single-tenant MVP.

Workspace identity is resolved from server configuration, never from
client-supplied input.  This prevents a client from selecting another
workspace by changing a query parameter or header.

Resolution order:
  1. ``WORKSPACE_ID`` environment variable
  2. ``TENANT_ID`` environment variable (legacy alias)
  3. ``"default-workspace"`` (single-tenant default)

The resolved value is returned as a :class:`uuid.UUID` so it is type-safe for
the ``workspace_id`` database column (``UUID(as_uuid=True)``).  A raw string
that is already a valid UUID is used directly; any other string is
deterministically mapped to a UUID via :func:`uuid.uuid5`, so the database
column always receives a valid UUID and never a raw string.

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
import uuid


_DEFAULT_WORKSPACE = "default-workspace"
_NAMESPACE = uuid.NAMESPACE_DNS


def resolve_workspace_id() -> uuid.UUID:
    """Return the server-resolved workspace identifier as a UUID.

    Reads environment configuration only.  Client input is never trusted,
    so a caller cannot access another workspace by supplying a different
    ``workspace_id``.

    Returns:
        A valid UUID.  Valid UUID strings are used directly; other strings
        (including the default ``"default-workspace"``) are deterministically
        mapped via :func:`uuid.uuid5`.

    Raises:
        ValueError: If the resolved raw value cannot be normalized to a UUID.
            (In practice ``uuid.uuid5`` accepts any string, so this only fires
            for degenerate inputs such as ``None``.)
    """
    raw = (
        os.environ.get("WORKSPACE_ID")
        or os.environ.get("TENANT_ID")
        or _DEFAULT_WORKSPACE
    )
    if raw is None:
        raise ValueError("Workspace identifier could not be resolved")

    # A valid UUID string is used directly (fails early with a clear error if
    # the configured value is malformed).  Any other string is deterministically
    # mapped to a UUID so the database column always receives a valid UUID.
    try:
        return uuid.UUID(str(raw))
    except (ValueError, AttributeError):
        return uuid.uuid5(_NAMESPACE, str(raw))


def workspace_id_str(workspace_id: uuid.UUID) -> str:
    """Return the string form of a workspace UUID for metadata storage.

    Chunk metadata (JSONB) and in-memory comparisons use the string form,
    while the database column uses the UUID form.
    """
    return str(workspace_id)
