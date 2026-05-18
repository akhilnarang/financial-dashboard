"""Email provider base helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from financial_dashboard.db import EmailSource

INITIAL_BACKFILL_DAYS = 90
JMAP_SESSION_URL = "https://api.fastmail.com/jmap/session"
FAILED_SPOOL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "failed"
FAILED_SPOOL_MAX_AGE_DAYS = 7


class EmailProvider(Protocol):
    async def fetch_source(
        self,
        source: EmailSource,
        rules,
        *,
        fetch_limit: int,
        existing_remote_ids: set[str],
    ) -> tuple[dict[int, list[tuple]], bool, set[int]]: ...

    async def fetch_single(
        self, source: EmailSource, remote_id: str
    ) -> bytes | None: ...


def get_provider(source: EmailSource) -> EmailProvider:
    # function-local: breaks cycle with integrations.email.imap_gmail and jmap_fastmail
    # (both providers import symbols from this module at top level).
    if source.provider == "gmail":
        from financial_dashboard.integrations.email.imap_gmail import GmailProvider

        return GmailProvider()
    if source.provider == "fastmail":
        from financial_dashboard.integrations.email.jmap_fastmail import (
            FastmailProvider,
        )

        return FastmailProvider()
    raise ValueError(f"unknown source kind: {source.provider}")
