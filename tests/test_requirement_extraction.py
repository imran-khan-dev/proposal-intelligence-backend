"""Unit tests for requirement extraction.

Uses representative fixtures and deterministic providers.  Covers numbered
requirements, mandatory/optional language, table requirements, nested
sections/lists, cross-references, multiple pages, duplicates, descriptive
prose, and source-reference integrity.  No external AI API is contacted.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import rfp_fixtures as fx
from app.modules.requirements.classification import (
    RequirementCategory,
    RequirementModality,
)
from app.modules.requirements.extraction import extract_requirements
from app.modules.requirements.parsing import StructuredTextParser


def _extract(pages, provider=None, doc_id="doc1"):
    doc = StructuredTextParser().parse(doc_id, "rfp.pdf", pages)
    return extract_requirements(doc, provider=provider)


class TestNumberedAndModality:
    def test_mandatory_and_optional_extracted(self):
        reqs = _extract(fx.numbered_fixture())
        modalities = {r.modality for r in reqs}
        assert RequirementModality.MANDATORY in modalities
        assert RequirementModality.OPTIONAL in modalities

    def test_original_text_preserved(self):
        reqs = _extract(fx.numbered_fixture())
        texts = " ".join(r.text for r in reqs)
        assert "must support SAML 2.0 single sign-on" in texts

    def test_section_title_and_path_attached(self):
        reqs = _extract(fx.numbered_fixture())
        # The SAML requirement lives under "Functional Requirements".
        saml = next(r for r in reqs if "SAML 2.0" in r.text)
        assert saml.section_title == "Functional Requirements"
        assert saml.section_path is not None


class TestTables:
    def test_requirements_inside_table_extracted(self):
        reqs = _extract(fx.table_fixture())
        texts = " ".join(r.text for r in reqs)
        assert "REQ-001" in texts
        assert "SAML 2.0" in texts
        assert "uptime SLA" in texts

    def test_table_header_not_a_requirement(self):
        reqs = _extract(fx.table_fixture())
        # The header row "ID | Requirement | Priority" must not be extracted.
        assert all("Priority" not in r.text for r in reqs)


class TestNestedListsAndCrossRefs:
    def test_bulleted_list_items_extracted(self):
        reqs = _extract(fx.nested_fixture())
        texts = " ".join(r.text for r in reqs)
        assert "Encryption of data at rest" in texts
        assert "Role-based access control" in texts

    def test_cross_reference_extracted(self):
        reqs = _extract(fx.nested_fixture())
        all_refs = [ref for r in reqs for ref in r.cross_references]
        assert any("Section 4.2" in ref for ref in all_refs)
        assert any("Attachment B" in ref for ref in all_refs)


class TestDuplicatesAndDescriptive:
    def test_duplicate_requirement_deduplicated(self):
        reqs = _extract(fx.duplicate_fixture())
        portal = [r for r in reqs if "cloud-based licensing portal" in r.text]
        assert len(portal) == 1

    def test_descriptive_text_not_extracted(self):
        reqs = _extract(fx.duplicate_fixture())
        texts = " ".join(r.text for r in reqs)
        assert "background of the project" not in texts
        assert "legacy software for over a decade" not in texts


class TestMultiplePagesAndSourceIntegrity:
    def test_multi_page_requirements_span_pages(self):
        reqs = _extract(fx.multi_page_fixture())
        pages = {r.page_number for r in reqs}
        assert len(pages) >= 2

    def test_source_offsets_map_to_original_text(self):
        pages = fx.numbered_fixture()
        full_text = "\n\n".join(p["text"] for p in pages)
        reqs = _extract(pages)
        assert reqs
        for r in reqs:
            span = full_text[r.character_start:r.character_end]
            # The requirement text must appear at its recorded span.
            assert r.text in span

    def test_stable_requirement_keys(self):
        pages = fx.numbered_fixture()
        first = [r.requirement_key for r in _extract(pages)]
        second = [r.requirement_key for r in _extract(pages)]
        assert first == second
        assert len(set(first)) == len(first)


class TestLLMIntegration:
    def test_llm_used_for_uncertain_candidate(self):
        from app.modules.requirements.classification import (
            LLMClassification,
            _ClassificationsResponse,
        )

        class _Provider:
            def __init__(self):
                self.calls = 0

            def generate_structured(self, system_prompt, user_text, response_model):
                self.calls += 1
                return _ClassificationsResponse(
                    items=[
                        LLMClassification(
                            is_requirement=True,
                            modality=RequirementModality.MANDATORY,
                            category=RequirementCategory.FUNCTIONAL,
                            confidence=0.9,
                            notes=["llm"],
                        )
                    ]
                )

            def chat_with_tools(self, messages, tools):
                raise NotImplementedError

        # A candidate with no rule signal but the LLM confirms it.
        pages = fx.make_pages(["The system provides a dashboard."])
        provider = _Provider()
        reqs = _extract(pages, provider=provider)
        assert provider.calls == 1
        assert len(reqs) == 1
        assert reqs[0].extraction_method in ("llm", "hybrid")
        assert reqs[0].category is RequirementCategory.FUNCTIONAL