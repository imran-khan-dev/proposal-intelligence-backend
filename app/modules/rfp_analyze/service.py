import os
from pypdf import PdfReader

from app.ai.factory import get_provider
from app.modules.rfp_analyze.schema import RFPAnalysis

SYSTEM_PROMPT = """You are an expert analyst specializing in Request for Proposal (RFP) documents.

Your task is to extract structured information from the provided RFP document.

Rules:
- Extract information ONLY from the provided RFP document. Do not invent or assume anything that is not explicitly stated.
- For scalar fields (title, client, deadline, budget), use null if the information is not present in the document.
- For list fields, use an empty array if no items are found.
- Preserve the original wording of items. Do not paraphrase or summarize individual requirements, deliverables, etc.
- Distinguish between the following categories:
  - requirements: Functional or technical needs that the responder must satisfy.
  - deliverables: Tangible outputs, documents, or work products the responder must produce.
  - evaluation_criteria: Specific scoring factors, rubrics, or criteria the evaluator will use to grade submissions.
  - submission_requirements: Procedural requirements for submitting the proposal (format, file types, contact information, submission portal, etc.).
- Return only a single JSON object matching the requested schema. No additional commentary or explanation."""


def extract_pdf_text(file_path: str) -> str:
    """Extract text from a PDF file, preserving page boundaries."""
    reader = PdfReader(file_path)

    if len(reader.pages) == 0:
        raise ValueError("PDF contains no pages")

    pages_text = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages_text.append(f"--- Page {page_number} ---\n{text}")

    full_text = "\n\n".join(pages_text)

    if not full_text.strip():
        raise ValueError("No extractable text found in PDF")

    return full_text


def analyze_text(rfp_text: str) -> RFPAnalysis:
    """Send RFP text to the configured AI provider and return a structured RFPAnalysis."""
    provider = get_provider()

    user_content = f"Analyze the following RFP document:\n\n{rfp_text}"

    return provider.generate_structured(
        system_prompt=SYSTEM_PROMPT,
        user_text=user_content,
        response_model=RFPAnalysis,
    )

def analyze_rfp(file_path: str) -> RFPAnalysis:
    """Extract text from a PDF and analyze it with the configured AI provider."""
    pdf_text = extract_pdf_text(file_path)
    return analyze_text(pdf_text)