"""
Exception hierarchy for the LLM module.
"""


class LLMError(Exception):
    """Base exception for all LLM-related errors."""


class LLMConfigurationError(LLMError):
    """Raised when LLM configuration is invalid or incomplete."""


class LLMRequestError(LLMError):
    """Raised when an LLM request contains invalid parameters."""


class LLMProviderError(LLMError):
    """Raised when a concrete LLM provider fails."""
    

class LLMGatewayError(LLMError):
    """Raised when the LLM Gateway cannot complete a request."""

class LLMResponseError(LLMError):
    """Raised when an LLM response is invalid or incomplete."""
