class EODHDError(Exception):
    """Base exception for the EODHD client."""


class EODHDHTTPError(EODHDError):
    def __init__(self, status_code, message=""):
        self.status_code = status_code
        self.message = message
        super().__init__(f"HTTP {status_code}: {message}")


class EODHDAuthError(EODHDHTTPError):
    """401/403 — invalid or missing api_token."""


class EODHDNotFoundError(EODHDHTTPError):
    """404 — symbol or resource not found."""


class EODHDRateLimitError(EODHDHTTPError):
    """429 — rate limit exceeded."""
