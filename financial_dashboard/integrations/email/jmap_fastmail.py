"""Fastmail JMAP provider."""

import asyncio
import datetime
import json
import logging
from dataclasses import dataclass
from typing import NamedTuple
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from financial_dashboard.core.crypto import decrypt_credentials
from financial_dashboard.db import FetchRule
from financial_dashboard.integrations.email.base import (
    INITIAL_BACKFILL_DAYS,
    JMAP_SESSION_URL,
    NETWORK_TIMEOUT_SECONDS,
    FetchedEmail,
    FetchSourceResult,
)
from financial_dashboard.services.rules import (
    _format_jmap_from_field,
    _matches_rule_filters,
)

logger = logging.getLogger(__name__)

JMAP_CORE = "urn:ietf:params:jmap:core"
JMAP_MAIL = "urn:ietf:params:jmap:mail"
JMAP_USING = [JMAP_CORE, JMAP_MAIL]
MIN_PAGE_SIZE = 50
REGULAR_SYNC_MARGIN_DAYS = 2
EMAIL_PROPERTIES = ["id", "blobId", "receivedAt", "from", "subject", "mailboxIds"]
BLOB_TYPE = "application/octet-stream"
BLOB_NAME = "email.eml"
RULE_SCOPED_METHOD_ERRORS = frozenset(
    {"invalidArguments", "unsupportedFilter", "unsupportedSort", "anchorNotFound"}
)


class JmapError(Exception):
    """Raised when the JMAP server refuses a call. It fails the whole source."""


class JmapRuleError(JmapError):
    """Raised when one rule cannot be searched. The other rules still run."""


class _Candidate(NamedTuple):
    remote_id: str
    blob_id: str


class _RuleFilter(NamedTuple):
    jmap_filter: dict[str, str]
    mailbox_id: str | None


@dataclass(frozen=True)
class JmapSession:
    """One authenticated Fastmail JMAP session.

    Attributes:
        token: The bearer token.
        api_url: The URL that takes method calls.
        account_id: The primary mail account.
        download_url: The URI template for blob downloads.
        max_objects_in_get: The server limit on ids in one ``/get``, if given.
    """

    token: str
    api_url: str
    account_id: str
    download_url: str
    max_objects_in_get: int | None

    def call(self, method_calls: list[list]) -> list[list]:
        """Send method calls in one request.

        Args:
            method_calls: The JMAP invocations, as ``[name, args, call_id]``.

        Returns:
            The ``methodResponses`` list, in call order.
        """
        body = {"using": JMAP_USING, "methodCalls": method_calls}
        return _jmap_request(self.token, self.api_url, body)["methodResponses"]

    def blob_url(self, blob_id: str) -> str:
        """Fill the download URI template for one blob.

        Args:
            blob_id: The JMAP blob id.

        Returns:
            The download URL, with each value percent-encoded.
        """
        values = {
            "accountId": self.account_id,
            "blobId": blob_id,
            "type": BLOB_TYPE,
            "name": BLOB_NAME,
        }
        url = self.download_url
        for key, value in values.items():
            url = url.replace("{" + key + "}", quote(value, safe=""))
        return url

    def download(self, blob_id: str) -> bytes:
        """Download one blob.

        Args:
            blob_id: The JMAP blob id.

        Returns:
            The raw blob bytes.
        """
        req = Request(
            self.blob_url(blob_id), headers={"Authorization": f"Bearer {self.token}"}
        )
        with urlopen(req, timeout=NETWORK_TIMEOUT_SECONDS) as resp:
            return resp.read()


def _jmap_request(token: str, url: str, body: dict | None = None) -> dict:
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if body:
        req = Request(
            url, data=json.dumps(body).encode(), headers=headers, method="POST"
        )
    else:
        req = Request(url, headers=headers)
    with urlopen(req, timeout=NETWORK_TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read())


def _require_https(url: str, field: str) -> str:
    """Refuse a session URL that would send the token in clear text."""
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.netloc:
        raise JmapError(f"JMAP session {field} is not https: {url}")
    return url


def open_jmap_session(token: str) -> JmapSession:
    """Fetch the Fastmail JMAP session.

    Args:
        token: The Fastmail API token.

    Returns:
        The session, with its URLs checked to be https.

    Raises:
        JmapError: The session has no mail account or a URL is not https.
    """
    session = _jmap_request(token, JMAP_SESSION_URL)
    account_id = session.get("primaryAccounts", {}).get(JMAP_MAIL)
    if not account_id:
        raise JmapError("JMAP session returned but no mail account found")
    core = session.get("capabilities", {}).get(JMAP_CORE, {})
    return JmapSession(
        token=token,
        api_url=_require_https(session["apiUrl"], "apiUrl"),
        account_id=account_id,
        download_url=_require_https(session["downloadUrl"], "downloadUrl"),
        max_objects_in_get=core.get("maxObjectsInGet"),
    )


def _method_args(responses: list[list], index: int, name: str) -> dict:
    """Return the arguments of one method response.

    An error response must not read as an empty result. An error that
    the rule's own arguments cause fails only that rule. Any other error,
    such as ``serverFail`` or ``accountNotFound``, fails the whole source.

    Args:
        responses: The ``methodResponses`` list.
        index: The position of the call in the request.
        name: The method name that the call used.

    Returns:
        The response arguments.

    Raises:
        JmapRuleError: The rule's arguments caused the error.
        JmapError: The server answered with another error or another method.
    """
    response_name, args, _call_id = responses[index]
    if response_name == "error":
        error_type = args.get("type")
        message = f"{name} failed: {error_type} {args.get('description') or ''}"
        if error_type in RULE_SCOPED_METHOD_ERRORS:
            raise JmapRuleError(message.strip())
        raise JmapError(message.strip())
    if response_name != name:
        raise JmapError(f"{name} answered as {response_name}")
    return args


def _resolve_mailbox_id(session: JmapSession, folder_name: str) -> str | None:
    """Find a mailbox id by name, exact match first, then case-insensitive."""
    responses = session.call(
        [
            [
                "Mailbox/query",
                {"accountId": session.account_id, "filter": {"name": folder_name}},
                "0",
            ],
            [
                "Mailbox/get",
                {
                    "accountId": session.account_id,
                    "#ids": {"resultOf": "0", "name": "Mailbox/query", "path": "/ids"},
                    "properties": ["id", "name"],
                },
                "1",
            ],
        ]
    )
    _method_args(responses, 0, "Mailbox/query")
    mailboxes = _method_args(responses, 1, "Mailbox/get")["list"]
    for mb in mailboxes:
        if mb["name"] == folder_name:
            return mb["id"]
    for mb in mailboxes:
        if mb["name"].lower() == folder_name.lower():
            return mb["id"]
    return None


def _after_date(rule: FetchRule, last_synced_at: datetime.datetime | None) -> str:
    """Return the JMAP ``after`` bound for one rule.

    A rule without a finished backfill, or a source never synced, scans
    the backfill window. Other rules scan from the last sync, with a margin.
    """
    if rule.initial_backfill_done_at is None or last_synced_at is None:
        since = datetime.datetime.now(datetime.UTC) - datetime.timedelta(
            days=INITIAL_BACKFILL_DAYS
        )
    else:
        since = last_synced_at - datetime.timedelta(days=REGULAR_SYNC_MARGIN_DAYS)
    return since.strftime("%Y-%m-%dT00:00:00Z")


def _rule_filter(
    session: JmapSession,
    rule: FetchRule,
    after: str,
    mailbox_cache: dict[str, str | None],
) -> _RuleFilter:
    """Build the Email/query filter for one rule.

    Raises:
        JmapRuleError: The rule's folder does not exist.
        JmapError: The folder lookup failed.
    """
    jmap_filter = {"after": after}
    mailbox_id = None
    if rule.folder:
        if rule.folder not in mailbox_cache:
            mailbox_cache[rule.folder] = _resolve_mailbox_id(session, rule.folder)
        if (mailbox_id := mailbox_cache[rule.folder]) is None:
            raise JmapRuleError(f"folder not found: {rule.folder}")
        jmap_filter["inMailbox"] = mailbox_id
    if rule.sender:
        jmap_filter["from"] = rule.sender
    if rule.subject:
        jmap_filter["subject"] = rule.subject
    return _RuleFilter(jmap_filter, mailbox_id)


def _email_matches(em: dict, rule: FetchRule, mailbox_id: str | None) -> bool:
    """Check one Email/get entry against the rule's folder and local filters."""
    if mailbox_id and mailbox_id not in em.get("mailboxIds", {}):
        return False
    return _matches_rule_filters(
        rule,
        sender=_format_jmap_from_field(em.get("from")),
        subject=em.get("subject", ""),
    )


def _rule_candidates(
    session: JmapSession,
    rule: FetchRule,
    rule_filter: _RuleFilter,
    fetch_limit: int,
) -> list[_Candidate]:
    """Page through the emails that match one rule, newest first.

    Email/get may answer in any order, so the query's id order sets the
    order. A server may lower the page limit, so its answered limit ends
    the paging.

    Raises:
        JmapRuleError: The rule's arguments caused an error.
        JmapError: A query or get call failed for another reason.
    """
    page_size = max(fetch_limit, MIN_PAGE_SIZE)
    if session.max_objects_in_get:
        page_size = min(page_size, session.max_objects_in_get)
    candidates: list[_Candidate] = []
    position = 0
    while fetch_limit <= 0 or len(candidates) < fetch_limit:
        responses = session.call(
            [
                [
                    "Email/query",
                    {
                        "accountId": session.account_id,
                        "filter": rule_filter.jmap_filter,
                        "sort": [{"property": "receivedAt", "isAscending": False}],
                        "position": position,
                        "limit": page_size,
                    },
                    "0",
                ],
                [
                    "Email/get",
                    {
                        "accountId": session.account_id,
                        "#ids": {
                            "resultOf": "0",
                            "name": "Email/query",
                            "path": "/ids",
                        },
                        "properties": EMAIL_PROPERTIES,
                    },
                    "1",
                ],
            ]
        )
        query = _method_args(responses, 0, "Email/query")
        by_id = {em["id"]: em for em in _method_args(responses, 1, "Email/get")["list"]}
        for email_id in query["ids"]:
            if fetch_limit > 0 and len(candidates) >= fetch_limit:
                break
            em = by_id.get(email_id)
            if em and _email_matches(em, rule, rule_filter.mailbox_id):
                candidates.append(_Candidate(email_id, em["blobId"]))
        position += len(query["ids"])
        if len(query["ids"]) < query.get("limit", page_size):
            break
    return candidates


def _fetch_fastmail_source_sync(
    rules: list[FetchRule],
    *,
    token: str,
    fetch_limit: int,
    source_id: int,
    existing_remote_ids: set[str],
    last_synced_at: datetime.datetime | None = None,
) -> FetchSourceResult:
    """Fetch emails from Fastmail via JMAP for all rules on one source.

    A rule that cannot be searched is skipped and keeps its backfill
    pending. A session, transport, download or server-level JMAP error
    fails the whole source.

    Args:
        rules: The enabled rules of the source.
        token: The Fastmail API token.
        fetch_limit: The most emails to take per rule; 0 or less means all.
        source_id: The email source id.
        existing_remote_ids: Remote ids already stored for this source.
        last_synced_at: When the source last synced, if ever.

    Returns:
        The new emails per rule, whether the fetch worked, and the rules
        whose backfill search finished.
    """
    results_by_rule: dict[int, list[FetchedEmail]] = {r.id: [] for r in rules}

    if not token:
        logger.warning("Fastmail token missing for source %s", source_id)
        return FetchSourceResult(results_by_rule, False, set())

    try:
        logger.info("Fastmail: Starting fetch for source %s", source_id)
        session = open_jmap_session(token)
        logger.info("Fastmail: Connected successfully for source %s", source_id)

        mailbox_cache: dict[str, str | None] = {}
        backfill_ready: set[int] = set()
        for rule in rules:
            try:
                rule_filter = _rule_filter(
                    session, rule, _after_date(rule, last_synced_at), mailbox_cache
                )
                candidates = _rule_candidates(session, rule, rule_filter, fetch_limit)
            except JmapRuleError as e:
                logger.error(
                    "Fastmail rule %s skipped for source %s: %s", rule.id, source_id, e
                )
                continue
            if rule.initial_backfill_done_at is None:
                backfill_ready.add(rule.id)
            for remote_id, blob_id in candidates:
                if remote_id in existing_remote_ids:
                    continue
                results_by_rule[rule.id].append(
                    FetchedEmail(
                        msg_id=f"src{source_id}:fastmail:{remote_id}",
                        remote_id=remote_id,
                        raw_bytes=session.download(blob_id),
                    )
                )
        return FetchSourceResult(results_by_rule, True, backfill_ready)
    except Exception as e:
        logger.error("Fastmail JMAP error for source %s: %s", source_id, e)
        return FetchSourceResult(results_by_rule, False, set())


def _fetch_fastmail_single_sync(token: str, remote_id: str) -> bytes | None:
    """Fetch a single email from Fastmail by JMAP email ID."""
    try:
        session = open_jmap_session(token)
        responses = session.call(
            [
                [
                    "Email/get",
                    {
                        "accountId": session.account_id,
                        "ids": [remote_id],
                        "properties": ["blobId"],
                    },
                    "0",
                ]
            ]
        )
        emails = _method_args(responses, 0, "Email/get")["list"]
        if not emails:
            return None
        return session.download(emails[0]["blobId"])
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
    ) -> FetchSourceResult:
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
