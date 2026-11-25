"""Representative RFP page fixtures for the requirements milestone.

Each fixture is a list of page dicts (``page_number``, ``text``,
``character_start``, ``character_end``) with absolute offsets computed exactly
like :func:`app.rag.ingest.extract_pdf_pages` (pages joined by ``"\\n\\n"``),
so they can be fed straight into the structured parser and requirement
extractor without any PDF or network dependency.
"""

from __future__ import annotations

from typing import List


def make_pages(page_texts: List[str]) -> List[dict]:
    """Build page dicts with absolute character offsets.

    Mirrors the offset convention used at ingestion: page ``N`` occupies
    ``[start, end)`` in the normalized text produced by joining page texts
    with ``"\\n\\n"``.
    """
    pages: List[dict] = []
    offset = 0
    for i, text in enumerate(page_texts, start=1):
        pages.append(
            {
                "page_number": i,
                "text": text,
                "character_start": offset,
                "character_end": offset + len(text),
            }
        )
        offset += len(text) + 2  # account for the "\n\n" separator
    return pages


# --- Numbered requirements + mandatory/optional language -------------------

NUMBERED_PAGE = (
    "1. Scope of Work\n"
    "The contractor shall provide a cloud-based licensing portal.\n"
    "2. Functional Requirements\n"
    "2.1 The system must support SAML 2.0 single sign-on.\n"
    "2.2 The system should provide a reporting dashboard.\n"
    "2.3 The system may support mobile access at the vendor's discretion.\n"
)

# --- Nested sections and lists + cross-references --------------------------

NESTED_PAGE = (
    "3. Technical Requirements\n"
    "3.1 Infrastructure\n"
    "The platform must achieve 99.9% uptime as described in Section 4.2.\n"
    "3.1.1 Encryption\n"
    "The vendor shall provide the following:\n"
    "- Encryption of data at rest and in transit\n"
    "- Role-based access control\n"
    "1. Multi-factor authentication\n"
    "2. Audit logging retained for one year\n"
    "The system must comply with the controls in Attachment B.\n"
)

# --- Requirements inside a table -------------------------------------------

TABLE_PAGE = (
    "4. Detailed Requirements\n"
    "ID | Requirement | Priority\n"
    "REQ-001 | The system must support SAML 2.0 single sign-on. | High\n"
    "REQ-002 | The system shall export the compliance matrix to CSV. | High\n"
    "REQ-003 | The vendor must provide a 99.9% uptime SLA. | Medium\n"
)

# --- Duplicate / repeated requirements + descriptive prose -----------------

DUPLICATE_PAGE = (
    "5. Additional Terms\n"
    "This RFP describes the background of the project for context only.\n"
    "The contractor shall provide a cloud-based licensing portal.\n"
    "The contractor shall provide a cloud-based licensing portal.\n"
    "The department has used legacy software for over a decade.\n"
)

# --- Empty / poorly extracted pages ----------------------------------------

EMPTY_PAGE = ""
POOR_PAGE = "ID"


def numbered_fixture() -> List[dict]:
    return make_pages([NUMBERED_PAGE])


def nested_fixture() -> List[dict]:
    return make_pages([NESTED_PAGE])


def table_fixture() -> List[dict]:
    return make_pages([TABLE_PAGE])


def duplicate_fixture() -> List[dict]:
    return make_pages([DUPLICATE_PAGE])


def multi_page_fixture() -> List[dict]:
    return make_pages([NUMBERED_PAGE, TABLE_PAGE, NESTED_PAGE])


def empty_pages_fixture() -> List[dict]:
    return make_pages([NUMBERED_PAGE, EMPTY_PAGE, POOR_PAGE])