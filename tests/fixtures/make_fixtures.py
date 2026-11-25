"""Generate reusable document fixtures for the M1 ingestion milestone.

Run with:
    python -m tests.fixtures.make_fixtures

Produces, under ``tests/fixtures/``:

- ``single_page.pdf``   -- ordinary one-page text PDF
- ``multi_page.pdf``    -- three-page text PDF
- ``tabular.pdf``       -- PDF with table-like (tabular) text
- ``sample.txt``        -- plain text document
- ``sample.md``         -- markdown document
- ``empty.txt``         -- empty document (invalid case)
- ``not_a_pdf.pdf``     -- invalid PDF bytes (invalid case)
"""

from __future__ import annotations

from pathlib import Path

from .pdfgen import build_pdf

FIXTURE_DIR = Path(__file__).resolve().parent

SINGLE_PAGE_TEXT = (
    "Request for Proposal: Statewide Licensing Portal\n"
    "Issued by: Department of Information Technology\n"
    "Deadline: 2026-12-15\n"
    "Budget: $2,500,000\n"
)

MULTI_PAGE_TEXT = [
    ["Request for Proposal: Statewide Licensing Portal", "Issued by: Department of Information Technology"],
    ["Section 1 - Scope of Work", "The contractor shall provide a cloud-based licensing portal.", "Deliverable: functional specification."],
    ["Section 2 - Evaluation Criteria", "Criteria: technical approach (40%), cost (30%), past performance (30%).", "Submission deadline: 2026-12-15."],
]

TABULAR_TEXT = [
    ["Requirement ID", "Description", "Priority"],
    ["REQ-001", "Support SAML 2.0 single sign-on.", "High"],
    ["REQ-002", "Export compliance matrix to CSV.", "High"],
    ["REQ-003", "Provide 99.9% uptime SLA.", "Medium"],
]

TXT_CONTENT = (
    "Company Security Policy\n"
    "-----------------------\n"
    "All vendors must comply with ISO 27001.\n"
    "Evidence of SOC 2 Type II attestation is required.\n"
)

MD_CONTENT = (
    "# Company Knowledge Base\n\n"
    "## Incident Response\n\n"
    "The incident response team follows NIST SP 800-61.\n\n"
    "## Data Retention\n\n"
    "Records are retained for seven (7) years.\n"
)


def write_text(filename: str, content: str) -> None:
    (FIXTURE_DIR / filename).write_text(content, encoding="utf-8")


def main() -> None:
    FIXTURE_DIR.joinpath("single_page.pdf").write_bytes(build_pdf([SINGLE_PAGE_TEXT.split("\n")]))
    FIXTURE_DIR.joinpath("multi_page.pdf").write_bytes(build_pdf(MULTI_PAGE_TEXT))
    FIXTURE_DIR.joinpath("tabular.pdf").write_bytes(build_pdf(TABULAR_TEXT))
    write_text("sample.txt", TXT_CONTENT)
    write_text("sample.md", MD_CONTENT)
    write_text("empty.txt", "")
    # Invalid PDF: not a real PDF.
    FIXTURE_DIR.joinpath("not_a_pdf.pdf").write_bytes(b"this is not a pdf file")


if __name__ == "__main__":
    main()
    print(f"Fixtures written to {FIXTURE_DIR}")
    for p in sorted(FIXTURE_DIR.iterdir()):
        print("  ", p.name, p.stat().st_size, "bytes")
