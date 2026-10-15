import os
import sys

# Clear the relevant environment variables
vars_to_clear = [
    'AI_TIMEOUT',
    'AI_MAX_RETRIES',
    'RFP_AI_PROVIDER',
    'AGENT_AI_PROVIDER',
    'RFP_AI_MAX_TOKENS',
    'AGENT_AI_MAX_TOKENS',
    'OLLAMA_MODEL',
    'OPENAI_API_KEY',
    'OPENAI_MODEL',
    'KILO_API_KEY',
    'KILO_MODEL',
    'GOOGLE_API_KEY',
    'GOOGLE_MODEL',
    'GROQ_API_KEY',
    'GROQ_MODEL'
]
for var in vars_to_clear:
    if var in os.environ:
        del os.environ[var]

# Set the environment variables for the test
os.environ['AI_TIMEOUT'] = '30.5'
os.environ['AI_MAX_RETRIES'] = '4'
os.environ['RFP_AI_PROVIDER'] = 'ollama'
os.environ['AGENT_AI_PROVIDER'] = 'openai'
os.environ['RFP_AI_MAX_TOKENS'] = '100'
os.environ['AGENT_AI_MAX_TOKENS'] = '200'
os.environ['OLLAMA_MODEL'] = 'test-ollama'
os.environ['OPENAI_API_KEY'] = 'test-openai-key'
os.environ['OPENAI_MODEL'] = 'test-openai-model'
os.environ['KILO_API_KEY'] = 'test-kilo-key'
os.environ['KILO_MODEL'] = 'test-kilo-model'
os.environ['GOOGLE_API_KEY'] = 'test-google-key'
os.environ['GOOGLE_MODEL'] = 'test-google-model'
os.environ['GROQ_API_KEY'] = 'test-groq-key'
os.environ['GROQ_MODEL'] = 'test-groq-model'

# Now import the factory module
sys.path.insert(0, 'E:\\proposal-intelligence\\backend')
from app.ai.factory import get_rfp_provider, get_agent_provider, get_provider
from app.ai.ollama import OllamaProvider
from app.ai.openai import OpenAIProvider
from app.ai.kilo import KiloProvider
from app.ai.google import GoogleProvider
from app.ai.groq import GroqProvider

# Get the providers
rfp_provider = get_rfp_provider()
agent_provider = get_agent_provider()
default_provider = get_provider()  # This should use AI_PROVIDER, which is not set, so defaults to openai

# Check the RFP provider (should be ollama)
print('RFP provider type:', type(rfp_provider).__name__)
assert isinstance(rfp_provider, OllamaProvider)
print('RFP provider timeout:', rfp_provider._timeout)
print('RFP provider max_retries:', rfp_provider._max_retries)
print('RFP provider max_tokens:', rfp_provider._max_tokens)
assert rfp_provider._timeout == 30.5
assert rfp_provider._max_retries == 4
assert rfp_provider._max_tokens == 100

# Check the agent provider (should be openai)
print('Agent provider type:', type(agent_provider).__name__)
assert isinstance(agent_provider, OpenAIProvider)
print('Agent provider timeout:', agent_provider._timeout)
print('Agent provider max_retries:', agent_provider._max_retries)
print('Agent provider max_tokens:', agent_provider._max_tokens)
assert agent_provider._timeout == 30.5
assert agent_provider._max_retries == 4
assert agent_provider._max_tokens == 200

# Check the default provider (should be openai because AI_PROVIDER is not set, and the factory defaults to openai)
print('Default provider type:', type(default_provider).__name__)
assert isinstance(default_provider, OpenAIProvider)
print('Default provider timeout:', default_provider._timeout)
print('Default provider max_retries:', default_provider._max_retries)
print('Default provider max_tokens:', default_provider._max_tokens)
# The default provider should use the global timeout and max_retries (since no task-specific max_tokens is passed)
assert default_provider._timeout == 30.5
assert default_provider._max_retries == 4
assert default_provider._max_tokens is None  # Because we didn't set RFP_AI_MAX_TOKENS or AGENT_AI_MAX_TOKENS for the default provider? Actually, the default provider uses get_provider() which calls _create_ai_provider_from_env with no prefix and max_tokens=None (the default). So max_tokens should be None.
assert default_provider._max_tokens is None

print('All tests passed!')
