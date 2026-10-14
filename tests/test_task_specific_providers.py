import os
import pytest
from unittest.mock import patch

from app.ai.factory import (
    get_rfp_provider,
    get_agent_provider,
    get_provider,
    _create_ai_provider_from_env
)


def test_get_provider_fallback_to_common():
    """Test that get_provider uses common AI_PROVIDER when no task-specific vars set."""
    with patch.dict(os.environ, {
        'AI_PROVIDER': 'openai',
        'OPENAI_API_KEY': 'test-key',
        'OPENAI_MODEL': 'gpt-test'
    }, clear=False):
        # Clear task-specific vars to ensure we're testing fallback
        if 'RFP_AI_PROVIDER' in os.environ:
            del os.environ['RFP_AI_PROVIDER']
        if 'AGENT_AI_PROVIDER' in os.environ:
            del os.environ['AGENT_AI_PROVIDER']
        
        provider = get_provider()
        # Should be an OpenAIProvider instance
        assert provider.__class__.__name__ == 'OpenAIProvider'


def test_get_rfp_provider_uses_rfp_vars_when_set():
    """Test that get_rfp_provider uses RFP_AI_PROVIDER when set."""
    with patch.dict(os.environ, {
        'RFP_AI_PROVIDER': 'groq',
        'GROQ_API_KEY': 'rfp-groq-key',
        'GROQ_MODEL': 'rfp-groq-model',
        # Also set common vars to ensure they're not used
        'AI_PROVIDER': 'openai',
        'OPENAI_API_KEY': 'common-openai-key',
        'OPENAI_MODEL': 'gpt-common'
    }):
        provider = get_rfp_provider()
        # Should be a GroqProvider instance
        assert provider.__class__.__name__ == 'GroqProvider'


def test_get_rfp_provider_fallback_to_common_when_empty():
    """Test that get_rfp_provider falls back to common vars when RFP_AI_PROVIDER is empty."""
    with patch.dict(os.environ, {
        'RFP_AI_PROVIDER': '',  # Empty string
        'AI_PROVIDER': 'google',
        'GOOGLE_API_KEY': 'rfp-google-key',
        'GOOGLE_MODEL': 'rfp-google-model',
        # Also set Groq vars to ensure they're not used
        'GROQ_API_KEY': 'should-not-be-used-key',
        'GROQ_MODEL': 'should-not-be-used-model'
    }):
        provider = get_rfp_provider()
        # Should be a GoogleProvider instance (fallback to common)
        assert provider.__class__.__name__ == 'GoogleProvider'


def test_get_agent_provider_uses_agent_vars_when_set():
    """Test that get_agent_provider uses AGENT_AI_PROVIDER when set."""
    with patch.dict(os.environ, {
        'AGENT_AI_PROVIDER': 'ollama',
        'OLLAMA_BASE_URL': 'http://agent-ollama:11434',
        'OLLAMA_MODEL': 'agent-ollama-model',
        # Also set common vars to ensure they're not used
        'AI_PROVIDER': 'openai',
        'OPENAI_API_KEY': 'common-openai-key',
        'OPENAI_MODEL': 'gpt-common'
    }):
        provider = get_agent_provider()
        # Should be an OllamaProvider instance
        assert provider.__class__.__name__ == 'OllamaProvider'


def test_get_agent_provider_fallback_to_common_when_empty():
    """Test that get_agent_provider falls back to common vars when AGENT_AI_PROVIDER is empty."""
    with patch.dict(os.environ, {
        'AGENT_AI_PROVIDER': '',  # Empty string
        'AI_PROVIDER': 'kilo',
        'KILO_API_KEY': 'agent-kilo-key',
        'KILO_MODEL': 'agent-kilo-model',
        'KILO_BASE_URL': 'https://agent.kilo.ai/api',
        # Also set other vars to ensure they're not used
        'GROQ_API_KEY': 'should-not-be-used-key',
        'GROQ_MODEL': 'should-not-be-used-model'
    }):
        provider = get_agent_provider()
        # Should be a KiloProvider instance (fallback to common)
        assert provider.__class__.__name__ == 'KiloProvider'


def test_both_providers_can_be_different():
    """Test that RFP and agent providers can be different when both task-specific vars are set."""
    with patch.dict(os.environ, {
        'RFP_AI_PROVIDER': 'groq',
        'GROQ_API_KEY': 'rfp-groq-key',
        'GROQ_MODEL': 'rfp-groq-model',
        'AGENT_AI_PROVIDER': 'openai',
        'OPENAI_API_KEY': 'agent-openai-key',
        'OPENAI_MODEL': 'agent-openai-model',
        # Also set common vars (should be ignored)
        'AI_PROVIDER': 'google',
        'GOOGLE_API_KEY': 'common-google-key',
        'GOOGLE_MODEL': 'common-google-model'
    }):
        rfp_provider = get_rfp_provider()
        agent_provider = get_agent_provider()
        
        # RFP provider should be Groq
        assert rfp_provider.__class__.__name__ == 'GroqProvider'
        
        # Agent provider should be OpenAI
        assert agent_provider.__class__.__name__ == 'OpenAIProvider'
        
        # They should be different instances
        assert rfp_provider is not agent_provider


def test_create_ai_provider_from_env_with_prefix():
    """Test the helper function directly."""
    with patch.dict(os.environ, {
        'TEST_AI_PROVIDER': 'ollama',
        'TEST_OLLAMA_BASE_URL': 'http://test.ollama:11434',
        'TEST_OLLAMA_MODEL': 'test-ollama-model'
    }):
        provider = _create_ai_provider_from_env('test')
        assert provider.__class__.__name__ == 'OllamaProvider'


def test_create_ai_provider_from_env_fallback():
    """Test that the helper function falls back to common vars."""
    with patch.dict(os.environ, {
        'TEST_AI_PROVIDER': '',  # Empty, should fallback
        'AI_PROVIDER': 'openai',
        'OPENAI_API_KEY': 'fallback-key',
        'OPENAI_MODEL': 'fallback-model'
    }):
        provider = _create_ai_provider_from_env('test')
        assert provider.__class__.__name__ == 'OpenAIProvider'
