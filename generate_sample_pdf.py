"""Generate a minimal valid PDF for testing RFP analysis."""
import zlib


def _escape_pdf_string(text: str) -> str:
    """Escape special characters for PDF string literals."""
    result = []
    for ch in text:
        if ch == "\\":
            result.append("\\\\")
        elif ch == "(":
            result.append("\\(")
        elif ch == ")":
            result.append("\\)")
        elif ch == "\r":
            result.append("\\r")
        elif ord(ch) < 0x20 or ord(ch) > 0x7e:
            result.append(f"\\{ord(ch):03o}")
        else:
            result.append(ch)
    return "".join(result)


def _make_content_stream(text: str) -> bytes:
    """Create a valid PDF content stream."""
    lines = []
    y = 750
    for line in text.split("\n"):
        escaped = _escape_pdf_string(line)
        lines.append(
            f"BT /F1 11 Tf 50 {y} Td ({escaped}) Tj ET".encode("latin-1")
        )
        y -= 14
        if y < 50:
            y = 750
    return b"\n".join(lines)


def create_sample_rfp(path: str) -> None:
    text = """RFP Number: 2026-001
Issuing Agency: Acme Corporation
Title: Customer Data Platform Implementation
Client: Acme Corporation
Deadline for Submission: November 15, 2026, 5:00 PM EST
Budget: $250,000

1. Overview
Acme Corporation seeks a Customer Data Platform (CDP) to unify customer data
across three business units and provide a single customer view.

2. Requirements
The vendor must deliver the following:
- Real-time customer data ingestion from Salesforce, HubSpot, and MySQL
- Identity resolution and profile unification across channels
  - Deterministic matching on email and phone
  - Stochastic matching on name and address
- GDPR-compliant data deletion and opt-out handling
- API access for downstream analytics tools
- Support for 10 million customer records

3. Deliverables
- Deployed CDP platform in AWS (us-east-1)
- Data migration for existing customer records
- API documentation and SDKs (Python and JavaScript)
- User training for data engineering team
- 12-month support and maintenance agreement

4. Evaluation Criteria
Proposals will be scored on the following:
- Technical approach and architecture (30%)
- Vendor experience and relevant case studies (20%)
- Cost and value proposition (20%)
- Project timeline and delivery methodology (15%)
- Team qualifications and staffing plan (15%)

5. Submission Requirements
- Submit proposals via Acme Procurement Portal by the deadline
- Maximum 25 pages, 12-point font, PDF format
- Include cover letter, technical proposal, and cost proposal
- One hard copy and one electronic copy required
- Contact: procurement@acmecorp.com, (555) 123-4567"""

    content = _make_content_stream(text)
    stream = zlib.compress(content)

    objects = []
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj")
    objects.append(
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj"
    )
    objects.append(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R "
        b"/MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> "
        b"/Contents 5 0 R >>\nendobj"
    )
    objects.append(
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj"
    )
    objects.append(
        f"5 0 obj\n<< /Length {len(stream)} /Filter /FlateDecode >>\nstream\n".encode("latin-1")
        + stream
        + b"\nendstream\nendobj"
    )

    pdf = b"%PDF-1.4\n"
    offsets = []
    for obj in objects:
        offsets.append(len(pdf))
        pdf += obj + b"\n"

    xref_pos = len(pdf)
    pdf += b"xref\n0 6\n"
    pdf += b"0000000000 65535 f \n"
    for off in offsets:
        pdf += f"{off:010d} 00000 n \n".encode("latin-1")
    pdf += b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n"
    pdf += f"{xref_pos}\n".encode("latin-1")
    pdf += b"%%EOF\n"

    with open(path, "wb") as f:
        f.write(pdf)


if __name__ == "__main__":
    create_sample_rfp("sample_rfp.pdf")
    print("Created sample_rfp.pdf")