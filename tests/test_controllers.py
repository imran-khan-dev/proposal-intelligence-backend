import pytest
import openai
from fastapi import HTTPException, UploadFile
from unittest.mock import AsyncMock, MagicMock, patch
import io

from app.modules.rfp_analyze.controller import analyze_rfp_controller
from app.modules.agent.controller import run_agent_controller
from app.modules.agent.schema import AgentRunRequest

def create_mock_response(status_code):
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.request = MagicMock()
    return mock_response

@pytest.mark.asyncio
async def test_rfp_controller_rate_limit_error():
    # Mock the provider to raise a RateLimitError
    from openai import RateLimitError
    mock_response = create_mock_response(429)
    exc = RateLimitError("Rate limit exceeded", response=mock_response, body=None)
    
    # Try to import the error handling function to see if there are any import errors
    try:
        from app.ai.error_handling import handle_ai_exception
    except Exception as e:
        print(f"Failed to import handle_ai_exception: {e}")
        raise
    
    # Mock the file reading to return non-empty bytes and the PDF text extraction to return a string
    with patch('app.modules.rfp_analyze.service.extract_pdf_text', return_value="dummy text"), \
         patch('app.modules.rfp_analyze.service.analyze_text', side_effect=exc):
        # Create a mock upload file
        file = MagicMock(spec=UploadFile)
        file.content_type = "application/pdf"
        # Simulate reading a file: first call returns the content, second call returns EOF
        file.read = AsyncMock(side_effect=[b"dummy pdf content", b""])
        file.close = AsyncMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await analyze_rfp_controller(file)
        
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail["error"]["code"] == "rate_limit_exceeded"
        assert exc_info.value.detail["error"]["retryable"] == True
        assert exc_info.value.detail["error"]["retry_after"] is None
        assert "Rate limit exceeded" in exc_info.value.detail["error"]["message"]
