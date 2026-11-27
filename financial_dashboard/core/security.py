"""HTTP Basic Auth for financial-dashboard."""

import secrets
from typing import Optional

from fastapi.security import HTTPBasic, HTTPBasicCredentials

from financial_dashboard.config import settings
from financial_dashboard.exceptions import UnauthorizedException

http_basic = HTTPBasic(auto_error=False)


def check_credentials(credentials: Optional[HTTPBasicCredentials]) -> None:
    if not settings.auth_enabled:
        return

    if credentials is None:
        raise UnauthorizedException(
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )

    username_ok = secrets.compare_digest(
        credentials.username.encode("utf-8"),
        settings.auth_username.encode("utf-8"),
    )
    password_ok = secrets.compare_digest(
        credentials.password.encode("utf-8"),
        settings.auth_password.get_secret_value().encode("utf-8"),
    )
    if not (username_ok and password_ok):
        raise UnauthorizedException(
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
