import pytest
import openai
import os
from fastapi import HTTPException, UploadFile
from unittest.mock import AsyncMock, MagicMock, patch
import io

from app.modules.rfp_analyze.controller import analyze_rfp_controller
from app.modules.agent.controller import run_agent_controller
from app.modules.agent.schema import AgentRunRequest

def create_mock_request():
    mock_request = MagicMock()
    return mock_request

def create_mock_response(status_code):
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.request = MagicMock()
    mock_response.headers = {}
    return mock_response

@pytest.mark.asyncio
async def test_rfp_controller_rate_limit_error():
    mock_response = create_mock_response(429)
    exc = openai.RateLimitError("Rate limit exceeded", response=mock_response, body=None)
    
    with patch('app.modules.rfp_analyze.service.extract_pdf_text', return_value="dummy text"), \
         patch('app.modules.rfp_analyze.service.analyze_text', side_effect=exc):
        file = MagicMock(spec=UploadFile)
        file.content_type = "application/pdf"
        file.read = AsyncMock(side_effect=[b"dummy pdf content", b""])
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
    mock_response = create_mock_response(401)
    exc = openai.AuthenticationError("Invalid API key", response=mock_response, body=None)
    mock_request = create_mock_request()
    exc.request = mock_request  # Set request attribute manually because constructor doesn't take it
    
    with patch('app.modules.rfp_analyze.service.extract_pdf_text', return_value="dummy text"), \
         patch('app.modules.rfp_analyze.service.analyze_text', side_effect=exc):
        file = MagicMock(spec=UploadFile)
        file.content_type = "application/pdf"
        file.read = AsyncMock(side_effect=[b"dummy pdf content", b""])
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
    exc = RuntimeError("Timeout occurred")
    
    with patch('app.modules.rfp_analyze.service.extract_pdf_text', return_value="dummy text"), \
         patch('app.modules.rfp_analyze.service.analyze_text', side_effect=exc):
        file = MagicMock(spec=UploadFile)
        file.content_type = "application/pdf"
        file.read = AsyncMock(side_effect=[b"dummy pdf content", b""])
        file.close = AsyncMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await analyze_rfp_controller(file)
        
        assert exc_info.value.status_code == 504
        assert exc_info.value.detail["error"]["code"] == "gateway_timeout"
        assert exc_info.value.detail["error"]["retryable"] == True
        assert exc_info.value.detail["error"]["retry_after"] is None
        assert "The AI provider did not respond in time." in exc_info.value.detail["error"]["message"]

@pytest.mark.asyncio
async def test_rfp_controller_service_unavailable():
    exc = RuntimeError("Service unavailable")
    
    with patch('app.modules.rfp_analyze.service.extract_pdf_text', return_value="dummy text"), \
         patch('app.modules.rfp_analyze.service.analyze_text', side_effect=exc):
        file = MagicMock(spec=UploadFile)
        file.content_type = "application/pdf"
        file.read = AsyncMock(side_effect=[b"dummy pdf content", b""])
        file.close = AsyncMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await analyze_rfp_controller(file)
        
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail["error"]["code"] == "service_unavailable"
        assert exc_info.value.detail["error"]["retryable"] == True
        assert exc_info.value.detail["error"]["retry_after"] is None
        assert "The AI provider is currently unavailable." in exc_info.value.detail["error"]["message"]

@pytest.mark.asyncio
async def test_rfp_controller_internal_error():
    exc = RuntimeError("Internal error")
    
    with patch('app.modules.rfp_analyze.service.extract_pdf_text', return_value="dummy text"), \
         patch('app.modules.rfp_analyze.service.analyze_text', side_effect=exc):
        file = MagicMock(spec=UploadFile)
        file.content_type = "application/pdf"
        file.read = AsyncMock(side_effect=[b"dummy pdf content", b""])
        file.close = AsyncMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await analyze_rfp_controller(file)
        
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail["error"]["code"] == "internal_error"
        assert exc_info.value.detail["error"]["retryable"] == False
        assert exc_info.value.detail["error"]["retry_after"] is None
        assert "The AI provider encountered an internal error." in exc_info.value.detail["error"]["message"]
