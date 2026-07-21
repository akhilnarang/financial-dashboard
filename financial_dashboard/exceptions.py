"""Named HTTP exceptions used by JSON and HTML route handlers."""

from typing import Any

from fastapi import HTTPException
from fastapi import status


class EmailParsePreviewError(Exception):
    """A sanitized email-preview failure and its intended HTTP status.

    Attributes:
        status_code: HTTP status the API wrapper should return.
    """

    def __init__(self, status_code: int, message: str) -> None:
        """Initialize a sanitized preview error.

        Args:
            status_code: HTTP status appropriate for the failure.
            message: Operator-safe error text; sensitive loader details must not
                be included.

        Returns:
            None.
        """
        super().__init__(message)
        self.status_code = status_code


class StatementPreviewError(Exception):
    """A sanitized statement-preview failure and its intended HTTP status.

    Attributes:
        status_code: HTTP status the API wrapper should return.
    """

    def __init__(self, status_code: int, message: str) -> None:
        """Initialize a sanitized statement-preview error.

        Args:
            status_code: HTTP status appropriate for the failure.
            message: Operator-safe error text that does not expose file paths,
                passwords, or parser internals.

        Returns:
            None.
        """
        super().__init__(message)
        self.status_code = status_code


class ApiException(HTTPException):
    """Base exception for an intentional HTTP error response."""

    status_code: int

    def __init__(
        self,
        *,
        detail: Any = None,
        headers: dict[str, str] | None = None,
        status_code: int | None = None,
    ):
        super().__init__(
            status_code=status_code or self.status_code,
            detail=detail,
            headers=headers,
        )


class BadRequestException(ApiException):
    """HTTP 400: the request is semantically invalid."""

    status_code = status.HTTP_400_BAD_REQUEST


class UnauthorizedException(ApiException):
    """HTTP 401: authentication is required or the credentials are invalid."""

    status_code = status.HTTP_401_UNAUTHORIZED


class NotFoundException(ApiException):
    """HTTP 404: the requested resource does not exist."""

    status_code = status.HTTP_404_NOT_FOUND


class ConflictException(ApiException):
    """HTTP 409: the request conflicts with current resource state."""

    status_code = status.HTTP_409_CONFLICT


class PayloadTooLargeException(ApiException):
    """HTTP 413: the request body exceeds an endpoint limit."""

    status_code = status.HTTP_413_CONTENT_TOO_LARGE


class UnprocessableEntityException(ApiException):
    """HTTP 422: validated input cannot be processed as requested."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT


class FailedDependencyException(ApiException):
    """HTTP 424: an external resource required by the request is unavailable."""

    status_code = status.HTTP_424_FAILED_DEPENDENCY


class InternalServerException(ApiException):
    """HTTP 500: an internal resource disappeared or became inconsistent."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
