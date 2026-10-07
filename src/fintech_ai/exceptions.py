"""Domain exception hierarchy. Callers catch FintechAIError at the boundary."""


class FintechAIError(Exception):
    """Base class for all application errors."""


class ConfigError(FintechAIError):
    """Invalid or missing configuration."""


class DataFetchError(FintechAIError):
    """Market data could not be retrieved or was unusable."""


class InsufficientDataError(FintechAIError):
    """Data was retrieved but has too few observations for the requested analysis."""


class LLMError(FintechAIError):
    """The language model call failed or returned an unusable response."""
