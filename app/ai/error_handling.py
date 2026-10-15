from typing import Optional
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

def handle_ai_exception(exc: Exception) -> HTTPException:
    """
    Convert an exception from an AI provider into an appropriate HTTPException.
    """
    # Default values
    status_code = 502
    detail = "AI provider request failed"
    error_code = "internal_error"
    retryable = False
    retry_after = None
    headers = None

    # Handle OpenAI-related exceptions (used by Ollama, OpenAI, Kilo, Groq)
    try:
        from openai import APIError
    except ImportError:
        APIError = None

    try:
        from openai import RateLimitError
    except ImportError:
        RateLimitError = None

    try:
        from openai import AuthenticationError
    except ImportError:
        AuthenticationError = None

    try:
        from openai import PermissionDeniedError
    except ImportError:
        PermissionDeniedError = None

    try:
        from openai import NotFoundError
    except ImportError:
        NotFoundError = None

    try:
        from openai import BadRequestError
    except ImportError:
        BadRequestError = None

    try:
        from openai import UnprocessableEntityError
    except ImportError:
        UnprocessableEntityError = None

    try:
        from openai import InternalServerError
    except ImportError:
        InternalServerError = None

    try:
        from openai import ServiceUnavailableError
    except ImportError:
        ServiceUnavailableError = None

    try:
        from openai import GatewayTimeoutError
    except ImportError:
        GatewayTimeoutError = None

    # Handle Google GenAI exceptions
    try:
        from google.genai import errors as genai_errors
        # We'll catch the base error and then check the type
        from google.genai.errors import APIError as GenAIAPIError
    except ImportError:
        genai_errors = None
        GenAIAPIError = None

    # Check for OpenAI exceptions
    if RateLimitError is not None and isinstance(exc, RateLimitError):
        status_code = 429
        error_code = "rate_limit_exceeded"
        retryable = True
        # Try to get retry-after from headers
        if hasattr(exc, 'response') and exc.response is not None:
            retry_after = exc.response.headers.get('retry-after')
            if retry_after:
                try:
                    retry_after = int(retry_after)
                except (ValueError, TypeError):
                    retry_after = None
        detail = "Rate limit exceeded. Please try again later."
    elif AuthenticationError is not None and isinstance(exc, AuthenticationError):
        status_code = 401
        error_code = "authentication_error"
        detail = "Invalid API key or authentication failed."
    elif PermissionDeniedError is not None and isinstance(exc, PermissionDeniedError):
        status_code = 403
        error_code = "permission_denied"
        detail = "Permission denied. Check your API key permissions."
    elif NotFoundError is not None and isinstance(exc, NotFoundError):
        status_code = 404
        error_code = "not_found"
        detail = "The requested resource was not found."
    elif BadRequestError is not None and isinstance(exc, BadRequestError):
        status_code = 400
        error_code = "bad_request"
        detail = "Invalid request. Please check your input."
    elif UnprocessableEntityError is not None and isinstance(exc, UnprocessableEntityError):
        status_code = 422
        error_code = "unprocessable_entity"
        detail = "The request was well-formed but was unable to be followed due to semantic errors."
    elif InternalServerError is not None and isinstance(exc, InternalServerError):
        status_code = 500
        error_code = "internal_error"
        detail = "The AI provider encountered an internal error."
    elif ServiceUnavailableError is not None and isinstance(exc, ServiceUnavailableError):
        status_code = 503
        error_code = "service_unavailable"
        retryable = True
        detail = "The AI provider is currently unavailable."
    elif GatewayTimeoutError is not None and isinstance(exc, GatewayTimeoutError):
        status_code = 504
        error_code = "gateway_timeout"
        retryable = True
        detail = "The AI provider did not respond in time."
    elif APIError is not None and isinstance(exc, APIError):
        # Map based on status code if available
        status_code = 502  # default
        error_code = "api_error"
        detail = f"AI provider error: {getattr(exc, 'message', str(exc))}"
        if hasattr(exc, 'response') and exc.response is not None:
            sc = exc.response.status_code
            if sc == 429:
                status_code = 429
                error_code = "rate_limit_exceeded"
                retryable = True
                detail = "Rate limit exceeded. Please try again later."
                # Try to get retry-after from headers
                if hasattr(exc.response, 'headers'):
                    retry_after = exc.response.headers.get('retry-after')
                    if retry_after:
                        try:
                            retry_after = int(retry_after)
                        except (ValueError, TypeError):
                            retry_after = None
            elif sc == 401:
                status_code = 401
                error_code = "authentication_error"
                detail = "Invalid API key or authentication failed."
            elif sc == 403:
                status_code = 403
                error_code = "permission_denied"
                detail = "Permission denied. Check your API key permissions."
            elif sc == 404:
                status_code = 404
                error_code = "not_found"
                detail = "The requested resource was not found."
            elif sc == 400:
                status_code = 400
                error_code = "bad_request"
                detail = "Invalid request. Please check your input."
            elif sc == 422:
                status_code = 422
                error_code = "unprocessable_entity"
                detail = "The request was well-formed but was unable to be followed due to semantic errors."
            elif sc == 500:
                status_code = 500
                error_code = "internal_error"
                detail = "The AI provider encountered an internal error."
            elif sc == 503:
                status_code = 503
                error_code = "service_unavailable"
                retryable = True
                detail = "The AI provider is currently unavailable."
            elif sc == 504:
                status_code = 504
                error_code = "gateway_timeout"
                retryable = True
                detail = "The AI provider did not respond in time."
            # else keep the default 502
    # Check for Google GenAI exceptions
    elif genai_errors is not None and GenAIAPIError is not None and isinstance(exc, GenAIAPIError):
        # Map Google GenAI errors to our error codes
        # We'll use the error's code if available, otherwise use a generic one
        error_code = getattr(exc, 'code', 'google_api_error')
        # Map common Google error codes to HTTP statuses
        # Ref: https://cloud.google.com/apis/design/errors
        if hasattr(exc, 'code'):
            if exc.code == 429:
                status_code = 429
                error_code = "rate_limit_exceeded"
                retryable = True
                detail = "Rate limit exceeded. Please try again later."
            elif exc.code == 401:
                status_code = 401
                error_code = "authentication_error"
                detail = "Invalid API key or authentication failed."
            elif exc.code == 403:
                status_code = 403
                error_code = "permission_denied"
                detail = "Permission denied. Check your API key permissions."
            elif exc.code == 400 or exc.code == 404:
                status_code = 400
                error_code = "bad_request"
                detail = "Invalid request. Please check your input."
            elif exc.code == 422:
                status_code = 422
                error_code = "unprocessable_entity"
                detail = "The request was well-formed but was unable to be followed due to semantic errors."
            elif exc.code == 500:
                status_code = 500
                error_code = "internal_error"
                detail = "The AI provider encountered an internal error."
            elif exc.code == 503:
                status_code = 503
                error_code = "service_unavailable"
                retryable = True
                detail = "The AI provider is currently unavailable."
            elif exc.code == 504:
                status_code = 504
                error_code = "gateway_timeout"
                retryable = True
                detail = "The AI provider did not respond in time."
            else:
                # For any other code, default to 502
                status_code = 502
                detail = f"Google AI provider error: {exc}"
        else:
            status_code = 502
            detail = f"Google AI provider error: {exc}"
    else:
        # If it's not one of the known exceptions, we'll treat it as a generic error.
        # But note: the providers might raise RuntimeError with a message, as we saw in the Google provider.
        # We'll check if the exception is a RuntimeError and if the message indicates a known issue.
        if isinstance(exc, RuntimeError):
            # We'll check the message for keywords to classify.
            msg = str(exc).lower()
            if "rate limit" in msg or "quota" in msg:
                status_code = 429
                error_code = "rate_limit_exceeded"
                retryable = True
                detail = "Rate limit exceeded. Please try again later."
            elif "authentication" in msg or "invalid key" in msg or "unauthorized" in msg:
                status_code = 401
                error_code = "authentication_error"
                detail = "Invalid API key or authentication failed."
            elif "timeout" in msg:
                status_code = 504
                error_code = "gateway_timeout"
                retryable = True
                detail = "The AI provider did not respond in time."
            elif "internal error" in msg:
                status_code = 500
                error_code = "internal_error"
                detail = "The AI provider encountered an internal error."
            elif "service unavailable" in msg or "unavailable" in msg:
                status_code = 503
                error_code = "service_unavailable"
                retryable = True
                detail = "The AI provider is currently unavailable."
            else:
                # Keep the original detail but sanitize
                detail = str(exc)
                # We don't want to expose stack traces, so we just use the exception message.
        else:
            # For any other exception, we'll return a 502 with the exception message.
            detail = str(exc)

    # If we have a retry_after, add it as a header
    if retry_after is not None:
        headers = {"Retry-After": str(retry_after)}

    # Log the exception for debugging (without sensitive info)
    logger.error(f"AI provider error: {error_code} - {detail}")

    return HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "code": error_code,
                "message": detail,
                "retryable": retryable,
                "retry_after": retry_after,
            }
        },
        headers=headers,
    )