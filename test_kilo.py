"""Test script for the Kilo provider and RFP analyzer."""
import os
import sys
import json

# Ensure we load .env
from dotenv import load_dotenv
load_dotenv()

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.ai.factory import get_provider
from app.ai.provider import AIProvider
from app.schemas.rfp import RFPAnalysis
from app.services.rfp_analyzer import analyze_text, extract_pdf_text


def test_provider_selection():
    print("=" * 60)
    print("TEST 1: Provider Selection")
    print("=" * 60)
    provider = get_provider()
    print(f"Provider type: {type(provider).__name__}")
    print(f"AI_PROVIDER env: {os.environ.get('AI_PROVIDER')}")
    assert isinstance(provider, AIProvider), "Provider must implement AIProvider"
    print("PASS: Provider implements AIProvider interface\n")


def test_simple_kilo_request():
    print("=" * 60)
    print("TEST 2: Simple Kilo Request")
    print("=" * 60)
    provider = get_provider()
    print(f"Using provider: {type(provider).__name__}")

    try:
        result = provider.generate_structured(
            system_prompt="You are a test assistant. Respond only with valid JSON matching the schema.",
            user_text='Say hello in one short sentence. Return JSON: {"greeting": "your message here"}',
            response_model=RFPAnalysis,
        )
        print(f"Response type: {type(result).__name__}")
        print(f"Response: {result.model_dump_json(indent=2)}")
        assert isinstance(result, RFPAnalysis), "Response must be RFPAnalysis"
        print("PASS: Kilo returned valid RFPAnalysis\n")
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}\n")
        raise


def test_rfp_analyzer_with_pdf():
    print("=" * 60)
    print("TEST 3: RFP Analyzer with Sample PDF")
    print("=" * 60)

    pdf_path = os.path.join(os.path.dirname(__file__), "sample_rfp.pdf")
    if not os.path.exists(pdf_path):
        print("Generating sample PDF...")
        from generate_sample_pdf import create_sample_rfp
        create_sample_rfp(pdf_path)

    print(f"PDF exists: {os.path.exists(pdf_path)}")

    # Test PDF extraction
    text = extract_pdf_text(pdf_path)
    print(f"Extracted text length: {len(text)} chars")
    print(f"First 200 chars: {text[:200]}")
    assert len(text) > 0, "PDF text extraction failed"
    print("PASS: PDF text extracted\n")

    # Test full analysis
    print("Calling analyze_text with Kilo provider...")
    result = analyze_text(text)
    print(f"Result type: {type(result).__name__}")
    print(f"Result: {result.model_dump_json(indent=2)}")
    assert isinstance(result, RFPAnalysis), "Result must be RFPAnalysis"
    print("PASS: RFP analysis returned valid RFPAnalysis\n")


def main():
    try:
        test_provider_selection()
        test_simple_kilo_request()
        test_rfp_analyzer_with_pdf()
        print("=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    except Exception as exc:
        print(f"\nTEST FAILED: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()