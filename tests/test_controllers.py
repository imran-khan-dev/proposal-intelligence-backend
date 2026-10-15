import pytest
from fastapi import HTTPException, UploadFile
from unittest.mock import AsyncMock, MagicMock, patch
import io

from app.modules.rfp_analyze.controller import analyze_rfp_controller
from app.modules.agent.controller import run_agent_controller
from app.modules.agent.schema import AgentRunRequest

@pytest.mark.asyncio
async def test_rfp_controller_rate_limit_error():
    # Mock the provider to raise a RateLimitError
    from openai import RateLimitError
    exc = RateLimitError("Rate limit exceeded")
    
    # Mock the file reading to return non-empty bytes and the PDF text extraction to return a string
    with patch('app.modules.rfp_analyze.service.extract_pdf_text', return_value="dummy text"), \
         patch('app.modules.rfp_analyze.service.analyze_text', side_effect=exc):
        # Create a mock upload file
        file = MagicMock(spec=UploadFile)
        file.content_type = "application/pdf"
        file.read = AsyncMock(return_value=b"dummy pdf content")  # Non-empty to avoid file read error
        file.close = AsyncMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await analyze_rfp_controller(file)
        
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail["error"]["code"] == "rate_limit_exceeded"
        assert exc_info.value.detail["error"]["retryable"] == True
        assert exc_info.value.detail["error"]["retry_after"] is None
        assert "Rate limit exceeded" in exc_info.value.detail["error"]["message"]

@pytest.mark.asyncio
async def test_rfp_controller_authentication_error():
    from openai import AuthenticationError
    exc = AuthenticationError("Invalid API key")
    
    with patch('app.modules.rfp_analyze.service.extract_pdf_text', return_value="dummy text"), \
         patch('app.modules.rfp_analyze.service.analyze_text', side_effect=exc):
        file = MagicMock(spec=UploadFile)
        file.content_type = "application/pdf"
        file.read = AsyncMock(return_value=b"dummy pdf content")
        file.close = AsyncMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await analyze_rfp_controller(file)
        
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"]["code"] == "authentication_error"
        assert exc_info.value.detail["error"]["retryable"] == False
        assert exc_info.value.detail["error"]["retry_after"] is None
        assert "Invalid API key" in exc_info.value.detail["error"]["message"]

@pytest.mark.asyncio
async def test_rfp_controller_timeout_error():
    # We'll simulate a timeout by raising a RuntimeError with a timeout message
    exc = RuntimeError("Timeout occurred")
    
    with patch('app.modules.rfp_analyze.service.extract_pdf_text', return_value="dummy text"), \
         patch('app.modules.rfp_analyze.service.analyze_text', side_effect=exc):
        file = MagicMock(spec=UploadFile)
        file.content_type = "application/pdf"
        file.read = AsyncMock(return_value=b"dummy pdf content")
        file.close = AsyncMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await analyze_rfp_controller(file)
        
        assert exc_info.value.status_code == 504
        assert exc_info.value.detail["error"]["code"] == "gateway_timeout"
        assert exc_info.value.detail["error"]["retryable"] == True
        assert exc_info.value.detail["error"]["retry_after"] is None
        assert "Timeout" in exc_info.value.detail["error"]["message"]

@pytest.mark.asyncio
async def test_rfp_controller_service_unavailable():
    # Note: ServiceUnavailableError might not be in the openai module in the version we have.
    # We'll use a generic APIError with status code 503.
    from openai import APIError
    # We'll create a mock response with status code 503
    mock_response = MagicMock()
    mock_response.status_code = 503
    exc = APIError("Service unavailable", response=mock_response, body=None)
    
    with patch('app.modules.rfp_analyze.service.extract_pdf_text', return_value="dummy text"), \
         patch('app.modules.rfp_analyze.service.analyze_text', side_effect=exc):
        file = MagicMock(spec=UploadFile)
        file.content_type = "application/pdf"
        file.read = AsyncMock(return_value=b"dummy pdf content")
        file.close = AsyncMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await analyze_rfp_controller(file)
        
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail["error"]["code"] == "service_unavailable"
        assert exc_info.value.detail["error"]["retryable"] == True
        assert exc_info.value.detail["error"]["retry_after"] is None
        assert "Service unavailable" in exc_info.value.detail["error"]["message"]

@pytest.mark.asyncio
async def test_rfp_controller_internal_error():
    from openai import InternalServerError
    exc = InternalServerError("Internal error")
    
    with patch('app.modules.rfp_analyze.service.extract_pdf_text', return_value="dummy text"), \
         patch('app.modules.rfp_analyze.service.analyze_text', side_effect=exc):
        file = MagicMock(spec=UploadFile)
        file.content_type = "application/pdf"
        file.read = AsyncMock(return_value=b"dummy pdf content")
        file.close = AsyncMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await analyze_rfp_controller(file)
        
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail["error"]["code"] == "internal_error"
        assert exc_info.value.detail["error"]["retryable"] == False
        assert exc_info.value.detail["error"]["retry_after"] is None
        assert "Internal error" in exc_info.value.detail["error"]["message"]

@pytest.mark.asyncio
async def test_agent_controller_rate_limit_error():
    from openai import RateLimitError
    exc = RateLimitError("Rate limit exceeded")
    
    with patch('app.ai.factory.get_agent_provider') as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.chat_with_tools = AsyncMock(side_effect=exc)
        mock_get_provider.return_value = mock_provider
        
        request = AgentRunRequest(message="Hello", max_tool_rounds=1)
        
        with pytest.raises(HTTPException) as exc_info:
            await run_agent_controller(request)
        
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail["error"]["code"] == "rate_limit_exceeded"
        assert exc_info.value.detail["error"]["retryable"] == True
        assert exc_info.value.detail["error"]["retry_after"] is None
        assert "Rate limit exceeded" in exc_info.value.detail["error"]["message"]

@pytest.mark.asyncio
async def test_agent_controller_authentication_error():
    from openai import AuthenticationError
    exc = AuthenticationError("Invalid API key")
    
    with patch('app.ai.factory.get_agent_provider') as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.chat_with_tools = AsyncMock(side_effect=exc)
        mock_get_provider.return_value = mock_provider
        
        request = AgentRunRequest(message="Hello", max_tool_rounds=1)
        
        with pytest.raises(HTTPException) as exc_info:
            await run_agent_controller(request)
        
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail["error"]["code"] == "authentication_error"
        assert exc_info.value.detail["error"]["retryable"] == False
        assert exc_info.value.detail["error"]["retry_after"] is None
        assert "Invalid API key" in exc_info.value.detail["error"]["message"]

@pytest.mark.asyncio
async def test_agent_controller_timeout_error():
    exc = RuntimeError("Timeout occurred")
    
    with patch('app.ai.factory.get_agent_provider') as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.chat_with_tools = AsyncMock(side_effect=exc)
        mock_get_provider.return_value = mock_provider
        
        request = AgentRunRequest(message="Hello", max_tool_rounds=1)
        
        with pytest.raises(HTTPException) as exc_info:
            await run_agent_controller(request)
        
        assert exc_info.value.status_code == 504
        assert exc_info.value.detail["error"]["code"] == "gateway_timeout"
        assert exc_info.value.detail["error"]["retryable"] == True
        assert exc_info.value.detail["error"]["retry_after"] is None
        assert "Timeout" in exc_info.value.detail["error"]["message"]

@pytest.mark.asyncio
async def test_agent_controller_service_unavailable():
    from openai import APIError
    mock_response = MagicMock()
    mock_response.status_code = 503
    exc = APIError("Service unavailable", response=mock_response, body=None)
    
    with patch('app.ai.factory.get_agent_provider') as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.chat_with_tools = AsyncMock(side_effect=exc)
        mock_get_provider.return_value = mock_provider
        
        request = AgentRunRequest(message="Hello", max_tool_rounds=1)
        
        with pytest.raises(HTTPException) as exc_info:
            await run_agent_controller(request)
        
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail["error"]["code"] == "service_unavailable"
        assert exc_info.value.detail["error"]["retryable"] == True
        assert exc_info.value.detail["error"]["retry_after"] is None
        assert "Service unavailable" in exc_info.value.detail["error"]["message"]

@pytest.mark.asyncio
async def test_agent_controller_internal_error():
    from openai import InternalServerError
    exc = InternalServerError("Internal error")
    
    with patch('app.ai.factory.get_agent_provider') as mock_get_provider:
        mock_provider = AsyncMock()
        mock_provider.chat_with_tools = AsyncMock(side_effect=exc)
        mock_get_provider.return_value = mock_provider
        
        request = AgentRunRequest(message="Hello", max_tool_rounds=1)
        
        with pytest.raises(HTTPException) as exc_info:
            await run_agent_controller(request)
        
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail["error"]["code"] == "internal_error"
        assert exc_info.value.detail["error"]["retryable"] == False
        assert exc_info.value.detail["error"]["retry_after"] is None
        assert "Internal error" in exc_info.value.detail["error"]["message"]
