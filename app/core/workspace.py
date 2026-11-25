"""Centralized workspace resolution for the single-tenant MVP.

Workspace identity is resolved from server configuration, never from
client-supplied input.  This prevents a client from selecting another
workspace by changing a query parameter or header.

Resolution order:
  1. ``WORKSPACE_ID`` environment variable (must be a valid UUID)
  2. ``TENANT_ID`` environment variable (legacy alias; must be a valid UUID)
  3. Fixed deterministic UUID (single-tenant default)

Explicit configuration that is not a valid UUID fails early with a clear
error instead of being silently converted into a different workspace.
"""

from __future__ import annotations

import os
import uuid

_FIXED_DEFAULT = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")


def _parse_uuid_env(name: str) -> uuid.UUID:
    """Parse a UUID environment variable, failing early if invalid."""
    raw = os.environ.get(name)
    if raw is None:
        raise KeyError(name)
    try:
        return uuid.UUID(raw)
    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"Environment variable {name} must be a valid UUID, got {raw!r}"
        ) from exc


def resolve_workspace_id() -> uuid.UUID:
    """Return the server-resolved workspace identifier as a UUID.

    Reads environment configuration only.  Client input is never trusted,
    so a caller cannot access another workspace by supplying a different
    ``workspace_id``.

    Returns:
        A valid UUID.  Explicit ``WORKSPACE_ID`` or ``TENANT_ID`` must be
        valid UUIDs; otherwise a fixed deterministic default is used.

    Raises:
        ValueError: If ``WORKSPACE_ID`` or ``TENANT_ID`` is set but is not
            a valid UUID.
    """
    try:
        return _parse_uuid_env("WORKSPACE_ID")
    except KeyError:
        pass
    try:
        return _parse_uuid_env("TENANT_ID")
    except KeyError:
        pass
    return _FIXED_DEFAULT


def workspace_id_str(workspace_id: uuid.UUID) -> str:
    """Return the string form of a workspace UUID for metadata storage.

    Chunk metadata (JSONB) and in-memory comparisons use the string form,
    while the database column uses the UUID form.
    """
    return str(workspace_id)
