"""LLM provider implementations with shared retry logic."""
import asyncio
import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)

# Retryable HTTP status codes (rate limit, server errors)
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds


def retry_on_transient(max_retries: int = _MAX_RETRIES, base_delay: float = _BASE_DELAY):
    """Decorator for sync functions: retry on transient HTTP/network errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    status = getattr(getattr(e, 'response', None), 'status_code', None)
                    is_retryable = (
                        status in _RETRYABLE_STATUS
                        or isinstance(e, (ConnectionError, TimeoutError))
                        or (hasattr(e, '__class__') and 'Timeout' in type(e).__name__)
                    )
                    if not is_retryable or attempt == max_retries:
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "Retry %d/%d for %s (status=%s): waiting %.1fs",
                        attempt + 1, max_retries, func.__name__, status, delay
                    )
                    time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator


def async_retry_on_transient(max_retries: int = _MAX_RETRIES, base_delay: float = _BASE_DELAY):
    """Decorator for async functions: retry on transient HTTP/network errors."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    status = getattr(getattr(e, 'response', None), 'status_code', None)
                    is_retryable = (
                        status in _RETRYABLE_STATUS
                        or isinstance(e, (ConnectionError, TimeoutError))
                        or (hasattr(e, '__class__') and 'Timeout' in type(e).__name__)
                    )
                    if not is_retryable or attempt == max_retries:
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "Retry %d/%d for %s (status=%s): waiting %.1fs",
                        attempt + 1, max_retries, func.__name__, status, delay
                    )
                    await asyncio.sleep(delay)
            raise last_exc
        return wrapper
    return decorator
