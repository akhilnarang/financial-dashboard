"""Email source service helpers."""

from __future__ import annotations

import asyncio
import imaplib
import json
from urllib.request import Request, urlopen

from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.core.crypto import decrypt_credentials
from financial_dashboard.db import EmailSource
from financial_dashboard.schemas.sources import SourceTestResponse

JMAP_SESSION_URL = "https://api.fastmail.com/jmap/session"


def _test_gmail(user: str, password: str) -> str:
    conn = imaplib.IMAP4_SSL("imap.gmail.com")
    conn.login(user, password)
    conn.logout()
    return f"Gmail IMAP login successful for {user}"


def _test_fastmail(token: str) -> str:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    req = Request(JMAP_SESSION_URL, headers=headers)
    with urlopen(req) as resp:
        data = json.loads(resp.read())
    acct = data.get("primaryAccounts", {}).get("urn:ietf:params:jmap:mail")
    if not acct:
        raise ValueError("JMAP session returned but no mail account found")
    return f"Fastmail JMAP session OK (account: {acct})"


class SourceNotFoundError(Exception):
    """Raised when a source_id doesn't resolve to a row."""


async def test_source_connectivity(
    session: AsyncSession,
    source_id: int,
) -> SourceTestResponse:
    source = await session.get(EmailSource, source_id)
    if not source:
        raise SourceNotFoundError(f"Source {source_id} not found")
    provider = source.provider
    encrypted_credentials = source.credentials

    try:
        creds = decrypt_credentials(encrypted_credentials)
    except Exception as exc:
        return SourceTestResponse(ok=False, error=f"Decryption failed: {exc}")

    if provider == "gmail":
        user = creds.get("user", "")
        password = creds.get("app_password", "")
        if not user or not password:
            return SourceTestResponse(
                ok=False, error="Missing user or app_password in credentials"
            )
        try:
            message = await asyncio.to_thread(_test_gmail, user, password)
            return SourceTestResponse(ok=True, message=message)
        except Exception as exc:
            return SourceTestResponse(ok=False, error=f"Gmail test failed: {exc}")

    if provider == "fastmail":
        token = creds.get("token", "")
        if not token:
            return SourceTestResponse(ok=False, error="Missing token in credentials")
        try:
            message = await asyncio.to_thread(_test_fastmail, token)
            return SourceTestResponse(ok=True, message=message)
        except Exception as exc:
            return SourceTestResponse(ok=False, error=f"Fastmail test failed: {exc}")

    return SourceTestResponse(ok=False, error=f"Unknown provider: {provider}")
