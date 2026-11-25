"""Unit tests for hybrid requirement classification.

Rule-based classification is deterministic and needs no model.  LLM-assisted
classification is tested with a deterministic fake provider, including
malformed output and provider failures.  No external AI API is contacted.
"""

from __future__ import annotations

from typing import Type, TypeVar

from pydantic import BaseModel

from app.modules.requirements.classification import (
    RequirementCategory,
    RequirementModality,
    RuleClassification,
    classify_by_rules,
    classify_with_llm,
)

T = TypeVar("T", bound=BaseModel)


class _FakeProvider:
    """Deterministic provider returning a preset structured response."""

    def __init__(self, response=None, raises: Exception | None = None):
        self._response = response
        self._raises = raises
        self.calls = 0

    def generate_structured(self, system_prompt, user_text, response_model):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._response

    def chat_with_tools(self, messages, tools):
        raise NotImplementedError


def _items(n, modality=RequirementModality.MANDATORY,
           category=RequirementCategory.SECURITY, is_req=True):
    from app.modules.requirements.classification import LLMClassification, _ClassificationsResponse

    return _ClassificationsResponse(
        items=[
            LLMClassification(
                is_requirement=is_req,
                modality=modality,
                category=category,
                confidence=0.88,
                notes=["llm"],
            )
            for _ in range(n)
        ]
    )


class TestRuleBasedModality:
    def test_must_is_mandatory(self):
        r = classify_by_rules("The system must support SAML 2.0.")
        assert r.modality is RequirementModality.MANDATORY
        assert r.is_requirement is True

    def test_shall_is_mandatory(self):
        r = classify_by_rules("The contractor shall provide a portal.")
        assert r.modality is RequirementModality.MANDATORY

    def test_should_is_optional(self):
        r = classify_by_rules("The system should provide a dashboard.")
        assert r.modality is RequirementModality.OPTIONAL

    def test_may_is_optional(self):
        r = classify_by_rules("The system may support mobile access.")
        assert r.modality is RequirementModality.OPTIONAL

    def test_no_signal_is_uncertain_not_requirement(self):
        r = classify_by_rules("The department has used legacy software.")
        assert r.modality is RequirementModality.UNCERTAIN
        assert r.is_requirement is False

    def test_conflicting_signals_preserve_uncertainty(self):
        r = classify_by_rules("The system must work but may be optional.")
        assert r.modality is RequirementModality.UNCERTAIN
        assert any("Conflicting" in n for n in r.notes)


class TestRuleBasedCategory:
    def test_security_category(self):
        r = classify_by_rules("The system must encrypt data and use SAML SSO.")
        assert r.category is RequirementCategory.SECURITY

    def test_submission_category(self):
        r = classify_by_rules("The proposal must be submitted via the portal.")
        assert r.category is RequirementCategory.SUBMISSION

    def test_support_category(self):
        r = classify_by_rules("The vendor shall provide 24/7 help desk support.")
        assert r.category is RequirementCategory.SUPPORT_MAINTENANCE

    def test_unknown_category_is_uncertain(self):
        r = classify_by_rules("The system must do the thing.")
        assert r.category is RequirementCategory.UNCERTAIN


class TestLLMFallback:
    def test_llm_classifies_uncertain_candidates(self):
        provider = _FakeProvider(response=_items(1))
        results = classify_with_llm(
            ["The system must support SAML."], provider
        )
        assert len(results) == 1
        assert results[0].category is RequirementCategory.SECURITY
        assert provider.calls == 1

    def test_no_provider_returns_none(self):
        results = classify_with_llm(["text"], None)
        assert results == [None]

    def test_provider_failure_returns_none(self):
        provider = _FakeProvider(raises=RuntimeError("boom"))
        results = classify_with_llm(["text"], provider)
        assert results == [None]

    def test_malformed_length_mismatch_returns_none(self):
        # Provider returns 1 item for 2 candidates -> mismatch -> all None.
        provider = _FakeProvider(response=_items(1))
        results = classify_with_llm(["a", "b"], provider)
        assert results == [None, None]

    def test_empty_candidates_no_call(self):
        provider = _FakeProvider(response=_items(0))
        results = classify_with_llm([], provider)
        assert results == []
        assert provider.calls == 0