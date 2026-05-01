from .client import EODHDClient
from .exceptions import (
    EODHDError,
    EODHDAuthError,
    EODHDRateLimitError,
    EODHDNotFoundError,
    EODHDHTTPError,
)

__all__ = [
    "EODHDClient",
    "EODHDError",
    "EODHDAuthError",
    "EODHDRateLimitError",
    "EODHDNotFoundError",
    "EODHDHTTPError",
]

__version__ = "0.1.0"
