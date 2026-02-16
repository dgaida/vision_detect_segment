import logging
from contextlib import contextmanager
from typing import Generator, Optional

from .exceptions import handle_redis_error


@contextmanager
def redis_operation(
    operation: str, host: str, port: int, logger: Optional[logging.Logger] = None, raise_on_error: bool = False
) -> Generator[None, None, None]:
    """
    Context manager for Redis operations with consistent error handling.

    Args:
        operation: Name of the operation being performed
        host: Redis host
        port: Redis port
        logger: Optional logger to log errors
        raise_on_error: Whether to re-raise the error after handling

    Yields:
        None

    Raises:
        RedisConnectionError: If raise_on_error is True and a Redis error occurs
    """
    try:
        yield
    except Exception as e:
        error = handle_redis_error(operation, host, port, e)
        if logger:
            logger.warning(str(error))
        if raise_on_error:
            raise error
