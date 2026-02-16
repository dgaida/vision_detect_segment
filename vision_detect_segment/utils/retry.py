import time
import functools
from typing import Callable, Any, Tuple, Type

def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay between attempts in seconds
        backoff_factor: Multiplier for the delay after each failed attempt
        exceptions: Tuple of exceptions to catch and retry on
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        raise

            if last_exception:
                raise last_exception
        return wrapper
    return decorator
