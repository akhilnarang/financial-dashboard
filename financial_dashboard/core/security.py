"""HTTP Basic Auth for financial-dashboard."""

import ipaddress
import secrets
from functools import lru_cache
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.status import HTTP_401_UNAUTHORIZED

from financial_dashboard.config import settings

http_basic = HTTPBasic(auto_error=False)


@lru_cache
def _get_trusted_networks() -> tuple[
    ipaddress.IPv4Network | ipaddress.IPv6Network, ...
]:
    networks = []
    for cidr in (s.strip() for s in settings.auth_skip_cidrs.split(",") if s.strip()):
        try:
            networks.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            raise SystemExit(f"Invalid CIDR in AUTH_SKIP_CIDRS: {cidr!r}")
    return tuple(networks)


def _is_trusted(client_host: str | None) -> bool:
    if not client_host:
        return False
    trusted = _get_trusted_networks()
    if not trusted:
        return False
    try:
        addr = ipaddress.ip_address(client_host)
    except ValueError:
        return False
    return any(addr in net for net in trusted)


def check_credentials(
    credentials: Optional[HTTPBasicCredentials],
    request: Optional[Request] = None,
) -> None:
    if not settings.auth_enabled:
        return

    if request and _is_trusted(request.client.host if request.client else None):
        return

    if credentials is None:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
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
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
