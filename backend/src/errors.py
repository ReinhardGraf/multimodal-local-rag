"""
Error response model for the backend API.
"""

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Structured error response model."""

    error: str
    detail: str | None = None
