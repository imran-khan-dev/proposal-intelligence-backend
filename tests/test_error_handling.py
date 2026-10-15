"""Direct tests for app.ai.error_handling.handle_ai_exception.

Covers OpenAI exception mapping, APIError catch-all status-code mapping,
runtime-error keyword classification, retry-after header parsing, and
the generic fallback path. No network calls are made.
"""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.ai.error_handling import handle_ai_exception


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(status_code=502, headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.request = MagicMock()
    return resp


def _openai_error(cls, message, status_code, headers=None):
    return cls(message, response=_mock_response(status_code, headers), body=None)


# ---------------------------------------------------------------------------
# 1. OpenAI RateLimitError
# ---------------------------------------------------------------------------

class TestRateLimitError:
    def test_without_retry_after_header(self):
        exc = _openai_error(__import__("openai").RateLimitError, "Rate limit exceeded", 429)
        result = handle_ai_exception(exc)
        assert isinstance(result, HTTPException)
        assert result.status_code == 429
        assert result.detail["error"]["code"] == "rate_limit_exceeded"
        assert result.detail["error"]["retryable"] is True
        assert result.detail["error"]["retry_after"] is None
        assert result.headers is None

    def test_with_retry_after_header(self):
        exc = _openai_error(
            __import__("openai").RateLimitError,
            "Rate limit exceeded",
            429,
            headers={"retry-after": "60"},
        )
        result = handle_ai_exception(exc)
        assert result.status_code == 429
        assert result.detail["error"]["retry_after"] == 60
        assert result.headers == {"Retry-After": "60"}

    def test_with_invalid_retry_after_header(self):
        exc = _openai_error(
            __import__("openai").RateLimitError,
            "Rate limit exceeded",
            429,
            headers={"retry-after": "not-a-number"},
        )
        result = handle_ai_exception(exc)
        assert result.status_code == 429
        assert result.detail["error"]["retry_after"] is None


# ---------------------------------------------------------------------------
# 2. OpenAI AuthenticationError
# ---------------------------------------------------------------------------

def test_authentication_error():
    exc = _openai_error(__import__("openai").AuthenticationError, "Invalid API key", 401)
    result = handle_ai_exception(exc)
    assert result.status_code == 401
    assert result.detail["error"]["code"] == "authentication_error"
    assert result.detail["error"]["retryable"] is False
    assert "Invalid API key" in result.detail["error"]["message"]


# ---------------------------------------------------------------------------
# 3. OpenAI PermissionDeniedError
# ---------------------------------------------------------------------------

def test_permission_denied_error():
    exc = _openai_error(__import__("openai").PermissionDeniedError, "Permission denied", 403)
    result = handle_ai_exception(exc)
    assert result.status_code == 403
    assert result.detail["error"]["code"] == "permission_denied"
    assert result.detail["error"]["retryable"] is False


# ---------------------------------------------------------------------------
# 4. OpenAI NotFoundError
# ---------------------------------------------------------------------------

def test_not_found_error():
    exc = _openai_error(__import__("openai").NotFoundError, "Resource not found", 404)
    result = handle_ai_exception(exc)
    assert result.status_code == 404
    assert result.detail["error"]["code"] == "not_found"


# ---------------------------------------------------------------------------
# 5. OpenAI BadRequestError
# ---------------------------------------------------------------------------

def test_bad_request_error():
    exc = _openai_error(__import__("openai").BadRequestError, "Bad input", 400)
    result = handle_ai_exception(exc)
    assert result.status_code == 400
    assert result.detail["error"]["code"] == "bad_request"


# ---------------------------------------------------------------------------
# 6. OpenAI UnprocessableEntityError
# ---------------------------------------------------------------------------

def test_unprocessable_entity_error():
    exc = _openai_error(__import__("openai").UnprocessableEntityError, "Unprocessable", 422)
    result = handle_ai_exception(exc)
    assert result.status_code == 422
    assert result.detail["error"]["code"] == "unprocessable_entity"


# ---------------------------------------------------------------------------
# 7. OpenAI InternalServerError
# ---------------------------------------------------------------------------

def test_internal_server_error():
    exc = _openai_error(__import__("openai").InternalServerError, "Server error", 500)
    result = handle_ai_exception(exc)
    assert result.status_code == 500
    assert result.detail["error"]["code"] == "internal_error"
    assert result.detail["error"]["retryable"] is False


# ---------------------------------------------------------------------------
# 8. OpenAI APIError catch-all (via APIStatusError with unmapped status)
# ---------------------------------------------------------------------------

class TestAPIErrorCatchAll:
    def test_429_mapped_with_retry_after(self):
        import openai as _o
        exc = _o.APIStatusError(
            "Too many requests",
            response=_mock_response(429, {"retry-after": "30"}),
            body=None,
        )
        result = handle_ai_exception(exc)
        assert result.status_code == 429
        assert result.detail["error"]["code"] == "rate_limit_exceeded"
        assert result.detail["error"]["retryable"] is True
        assert result.detail["error"]["retry_after"] == 30
        assert result.headers == {"Retry-After": "30"}

    def test_401_mapped(self):
        import openai as _o
        exc = _o.APIStatusError(
            "Unauthorized",
            response=_mock_response(401),
            body=None,
        )
        result = handle_ai_exception(exc)
        assert result.status_code == 401
        assert result.detail["error"]["code"] == "authentication_error"

    def test_503_mapped_retryable(self):
        import openai as _o
        exc = _o.APIStatusError(
            "Service unavailable",
            response=_mock_response(503),
            body=None,
        )
        result = handle_ai_exception(exc)
        assert result.status_code == 503
        assert result.detail["error"]["code"] == "service_unavailable"
        assert result.detail["error"]["retryable"] is True

    def test_502_default_when_unmapped(self):
        import openai as _o
        exc = _o.APIStatusError(
            "Bad gateway",
            response=_mock_response(502),
            body=None,
        )
        result = handle_ai_exception(exc)
        assert result.status_code == 502
        assert result.detail["error"]["code"] == "api_error"
        assert result.detail["error"]["retryable"] is False

    def test_no_response_attribute_defaults_to_502(self):
        import openai as _o
        exc = _o.APIError("Some error", request=MagicMock(), body=None)
        result = handle_ai_exception(exc)
        assert result.status_code == 502
        assert result.detail["error"]["code"] == "api_error"


# ---------------------------------------------------------------------------
# 9. RuntimeError keyword classification
# ---------------------------------------------------------------------------

class TestRuntimeErrorKeywords:
    @pytest.mark.parametrize(
        "message,expected_code,expected_status,expected_retryable",
        [
            ("Timeout occurred", "gateway_timeout", 504, True),
            ("Rate limit exceeded", "rate_limit_exceeded", 429, True),
            ("Quota exceeded", "rate_limit_exceeded", 429, True),
            ("invalid key", "authentication_error", 401, False),
            ("Unauthorized access", "authentication_error", 401, False),
            ("Internal error occurred", "internal_error", 500, False),
            ("Service unavailable", "service_unavailable", 503, True),
            ("Unknown error something", "internal_error", 502, False),
        ],
    )
    def test_keyword_classification(self, message, expected_code, expected_status, expected_retryable):
        exc = RuntimeError(message)
        result = handle_ai_exception(exc)
        assert result.status_code == expected_status
        assert result.detail["error"]["code"] == expected_code
        assert result.detail["error"]["retryable"] is expected_retryable

    def test_unknown_runtime_error_includes_original_message(self):
        exc = RuntimeError("Unexpected provider failure")
        result = handle_ai_exception(exc)
        assert result.status_code == 502
        assert "Unexpected provider failure" in result.detail["error"]["message"]


# ---------------------------------------------------------------------------
# 10. Generic Exception fallback
# ---------------------------------------------------------------------------

def test_generic_exception_returns_502():
    exc = ValueError("Something went wrong")
    result = handle_ai_exception(exc)
    assert result.status_code == 502
    assert result.detail["error"]["retryable"] is False
    assert result.detail["error"]["retry_after"] is None
    assert "Something went wrong" in result.detail["error"]["message"]
