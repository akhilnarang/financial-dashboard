"""Fastmail JMAP provider."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
from urllib.request import Request, urlopen

from financial_dashboard.core.crypto import decrypt_credentials
from financial_dashboard.integrations.email.base import (
    INITIAL_BACKFILL_DAYS,
    JMAP_SESSION_URL,
)
from financial_dashboard.services.rules import (
    _format_jmap_from_field,
    _matches_rule_filters,
)

logger = logging.getLogger(__name__)


def _jmap_request(token: str, url: str, body: dict | None = None) -> dict:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if body:
        req = Request(
            url, data=json.dumps(body).encode(), headers=headers, method="POST"
        )
    else:
        req = Request(url, headers=headers)
    with urlopen(req) as resp:
        return json.loads(resp.read())


def _resolve_mailbox_id(
    token: str, api_url: str, account_id: str, folder_name: str
) -> str | None:
    body = {
        "using": ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:mail"],
        "methodCalls": [
            [
                "Mailbox/query",
                {
                    "accountId": account_id,
                    "filter": {"name": folder_name},
                },
                "0",
            ],
            [
                "Mailbox/get",
                {
                    "accountId": account_id,
                    "#ids": {"resultOf": "0", "name": "Mailbox/query", "path": "/ids"},
                    "properties": ["id", "name"],
                },
                "1",
            ],
        ],
    }
    result = _jmap_request(token, api_url, body)
    mailboxes = result["methodResponses"][1][1].get("list", [])
    for mb in mailboxes:
        if mb["name"] == folder_name:
            return mb["id"]
    for mb in mailboxes:
        if mb["name"].lower() == folder_name.lower():
            return mb["id"]
    return None


def _fetch_fastmail_source_sync(
    rules,
    *,
    token: str,
    fetch_limit: int,
    source_id: int,
    existing_remote_ids: set[str],
    last_synced_at: datetime.datetime | None = None,
) -> tuple[dict[int, list[tuple]], bool, set[int]]:
    """Fetch emails from Fastmail via JMAP for all rules on one source.

    Establishes the JMAP session once, resolves mailbox IDs once, runs all
    rules' queries on the shared session. Returns
    ``(results_by_rule, fetch_ok, backfill_ready_rule_ids)``.
    """
    results_by_rule: dict[int, list[tuple]] = {r.id: [] for r in rules}

    if not token:
        logger.warning("Fastmail token missing for source %s", source_id)
        return results_by_rule, False, set()

    try:
        logger.info("Fastmail: Starting fetch for source %s", source_id)
        session = _jmap_request(token, JMAP_SESSION_URL)
        api_url = session["apiUrl"]
        account_id = session["primaryAccounts"]["urn:ietf:params:jmap:mail"]
        download_url = session["downloadUrl"]
        logger.info("Fastmail: Connected successfully for source %s", source_id)

        # Cache mailbox IDs by folder name
        mailbox_cache: dict[str, str | None] = {}

        # Date filter: 2-day margin for regular sync, 3 months for initial backfill
        after_date = None
        if last_synced_at is not None:
            since = last_synced_at - datetime.timedelta(days=2)
            after_date = since.strftime("%Y-%m-%dT00:00:00Z")
        else:
            # For initial backfill, use 3 months ago
            since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
                days=INITIAL_BACKFILL_DAYS
            )
            after_date = since.strftime("%Y-%m-%dT00:00:00Z")

        # Collect all candidate remote_ids per rule before downloading blobs
        # rule_id -> [(remote_id, blob_id, sender_text, subject_text, mailbox_id_filter), ...]
        rule_candidates: dict[int, list[tuple]] = {r.id: [] for r in rules}

        for rule in rules:
            mailbox_id = None
            jmap_filter: dict = {}

            if rule.folder:
                if rule.folder not in mailbox_cache:
                    mailbox_cache[rule.folder] = _resolve_mailbox_id(
                        token,
                        api_url,
                        account_id,
                        rule.folder,
                    )
                mailbox_id = mailbox_cache[rule.folder]
                if mailbox_id:
                    jmap_filter["inMailbox"] = mailbox_id
                else:
                    logger.error("Fastmail folder not found: %s", rule.folder)
                    continue
            if rule.sender:
                jmap_filter["from"] = rule.sender
            if rule.subject:
                jmap_filter["subject"] = rule.subject
            # For rules that haven't completed initial backfill, cap the
            # historical scan at 3 months rather than a full-history search.
            needs_backfill = getattr(rule, "initial_backfill_done_at", None) is None
            if needs_backfill:
                backfill_after = (
                    datetime.datetime.now(datetime.timezone.utc)
                    - datetime.timedelta(days=INITIAL_BACKFILL_DAYS)
                ).strftime("%Y-%m-%dT00:00:00Z")
                jmap_filter["after"] = backfill_after
                logger.info(
                    "Rule %s (bank=%s, sender=%s) initial backfill — after %s",
                    rule.id,
                    rule.bank,
                    rule.sender,
                    backfill_after,
                )
            elif after_date:
                jmap_filter["after"] = after_date

            position = 0
            page_size = max(fetch_limit, 50)
            rule_count = 0
            while fetch_limit <= 0 or rule_count < fetch_limit:
                body = {
                    "using": ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:mail"],
                    "methodCalls": [
                        [
                            "Email/query",
                            {
                                "accountId": account_id,
                                "filter": jmap_filter,
                                "sort": [
                                    {"property": "receivedAt", "isAscending": False}
                                ],
                                "position": position,
                                "limit": page_size,
                            },
                            "0",
                        ],
                        [
                            "Email/get",
                            {
                                "accountId": account_id,
                                "#ids": {
                                    "resultOf": "0",
                                    "name": "Email/query",
                                    "path": "/ids",
                                },
                                "properties": [
                                    "id",
                                    "blobId",
                                    "receivedAt",
                                    "from",
                                    "subject",
                                    "mailboxIds",
                                ],
                            },
                            "1",
                        ],
                    ],
                }

                result = _jmap_request(token, api_url, body)
                query_result = result["methodResponses"][0][1]
                emails = result["methodResponses"][1][1].get("list", [])
                query_ids = query_result.get("ids", [])
                if not query_ids:
                    break

                for em in emails:
                    if fetch_limit > 0 and rule_count >= fetch_limit:
                        break

                    remote_id = em["id"]
                    sender_text = _format_jmap_from_field(em.get("from"))
                    subject_text = em.get("subject", "")

                    if mailbox_id and mailbox_id not in em.get("mailboxIds", {}):
                        continue
                    if not _matches_rule_filters(
                        rule, sender=sender_text, subject=subject_text
                    ):
                        continue

                    rule_candidates[rule.id].append(
                        (
                            remote_id,
                            em["blobId"],
                            sender_text,
                            subject_text,
                        )
                    )
                    rule_count += 1

                position += len(query_ids)
                if len(query_ids) < page_size:
                    break

        # Scoped dedup: filter candidates against the caller-supplied set
        # Download blobs only for new emails
        for rule in rules:
            for remote_id, blob_id, sender_text, subject_text in rule_candidates[
                rule.id
            ]:
                if remote_id in existing_remote_ids:
                    continue

                msg_id = f"src{source_id}:fastmail:{remote_id}"

                blob_url = (
                    download_url.replace("{accountId}", account_id)
                    .replace("{blobId}", blob_id)
                    .replace("{type}", "application/octet-stream")
                    .replace("{name}", "email.eml")
                )
                req = Request(blob_url, headers={"Authorization": f"Bearer {token}"})
                with urlopen(req) as resp:
                    raw = resp.read()
                results_by_rule[rule.id].append((msg_id, remote_id, raw))

        # Build backfill-ready set: rules that got results or had
        # genuinely zero candidates.
        backfill_ready: set[int] = set()
        for rule in rules:
            if getattr(rule, "initial_backfill_done_at", None) is not None:
                continue  # already backfilled, skip
            if results_by_rule.get(rule.id):
                backfill_ready.add(rule.id)
            elif not rule_candidates.get(rule.id):
                # JMAP query returned zero candidates — genuinely empty
                backfill_ready.add(rule.id)
        return results_by_rule, True, backfill_ready
    except Exception as e:
        logger.error("Fastmail JMAP error for source %s: %s", source_id, e)
        return results_by_rule, False, set()


def _fetch_fastmail_single_sync(token: str, remote_id: str) -> bytes | None:
    """Fetch a single email from Fastmail by JMAP email ID."""
    try:
        session = _jmap_request(token, JMAP_SESSION_URL)
        api_url = session["apiUrl"]
        account_id = session["primaryAccounts"]["urn:ietf:params:jmap:mail"]
        download_url = session["downloadUrl"]

        body = {
            "using": ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:mail"],
            "methodCalls": [
                [
                    "Email/get",
                    {
                        "accountId": account_id,
                        "ids": [remote_id],
                        "properties": ["blobId"],
                    },
                    "0",
                ]
            ],
        }
        result = _jmap_request(token, api_url, body)
        emails = result["methodResponses"][0][1].get("list", [])
        if not emails:
            return None
        blob_url = (
            download_url.replace("{accountId}", account_id)
            .replace("{blobId}", emails[0]["blobId"])
            .replace("{type}", "application/octet-stream")
            .replace("{name}", "email.eml")
        )
        req = Request(blob_url, headers={"Authorization": f"Bearer {token}"})
        with urlopen(req) as resp:
            return resp.read()
    except Exception as e:
        logger.error("Failed to fetch Fastmail message %s: %s", remote_id, e)
        return None


class FastmailProvider:
    async def fetch_source(
        self,
        source,
        rules,
        *,
        fetch_limit: int,
        existing_remote_ids: set[str],
    ):
        creds = decrypt_credentials(source.credentials)
        return await asyncio.to_thread(
            _fetch_fastmail_source_sync,
            rules,
            token=creds.get("token", ""),
            fetch_limit=fetch_limit,
            source_id=source.id,
            existing_remote_ids=existing_remote_ids,
            last_synced_at=source.last_synced_at,
        )

    async def fetch_single(self, source, remote_id: str) -> bytes | None:
        creds = decrypt_credentials(source.credentials)
        return await asyncio.to_thread(
            _fetch_fastmail_single_sync, creds.get("token", ""), remote_id
        )
