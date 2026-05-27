"""Gmail IMAP provider."""

import asyncio
import datetime
import email as email_lib
import imaplib
import logging
import re
from collections import defaultdict
from typing import cast

from financial_dashboard.core.crypto import decrypt_credentials
from financial_dashboard.integrations.email.base import (
    INITIAL_BACKFILL_DAYS,
    FetchedEmail,
    FetchSourceResult,
)
from financial_dashboard.integrations.email.parsing import (
    _decode_header_value,
    _extract_message_metadata,
)
from financial_dashboard.services.rules import _matches_rule_filters

logger = logging.getLogger(__name__)


def _imap_since_date(last_synced_at: datetime.datetime | None) -> str | None:
    """Return IMAP SINCE date string with 2-day margin, or None."""
    if last_synced_at is None:
        # For initial backfill, use 3 months ago
        since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            days=INITIAL_BACKFILL_DAYS
        )
        return since.strftime("%d-%b-%Y")
    since = last_synced_at - datetime.timedelta(days=2)
    return since.strftime("%d-%b-%Y")


def _fetch_gmail_source_sync(
    rules,
    *,
    user: str,
    password: str,
    fetch_limit: int,
    source_id: int,
    existing_remote_ids: set[str],
    last_synced_at: datetime.datetime | None = None,
) -> FetchSourceResult:
    """Fetch emails from Gmail via IMAP for all rules on one source.

    Opens a single IMAP connection, runs SEARCH for each rule, deduplicates
    by X-GM-MSGID, does a two-phase fetch (metadata then RFC822), and returns
    a ``FetchSourceResult`` whose ``backfill_ready_rule_ids`` is the set of
    rule IDs whose Phase-0 SEARCH completed (even if it returned zero
    results). A rule whose search was skipped due to an IMAP error should
    **not** appear in this set, so the caller knows not to mark its backfill
    as complete.
    """
    if not user or not password:
        logger.warning("Gmail credentials missing for source %s", source_id)
        return FetchSourceResult({}, False, set())

    results_by_rule: dict[int, list[FetchedEmail]] = {r.id: [] for r in rules}
    # Track rule IDs whose Phase-0 SEARCH completed successfully (OK
    # response, even if zero results).  Only these should be candidates
    # for marking backfill complete.
    backfill_searched_rule_ids: set[int] = set()
    conn = imaplib.IMAP4_SSL("imap.gmail.com")
    try:
        logger.info("Gmail: Connecting to imap.gmail.com for source %s", source_id)
        conn.login(user, password)
        logger.info("Gmail: Logged in successfully for source %s", source_id)

        since_str = _imap_since_date(last_synced_at)

        # Phase 0: collect UIDs per rule, track which rules each UID belongs to
        uid_to_rules: dict[tuple, list] = {}  # (folder, uid) -> [rule, ...]
        current_folder = None

        for rule in rules:
            folder = rule.folder or "[Gmail]/All Mail"
            if folder != current_folder:
                typ, _ = conn.select(f'"{folder}"', readonly=True)
                if typ != "OK":
                    logger.error("Could not select Gmail folder: %s", folder)
                    continue
                current_folder = folder

            # For rules that haven't completed their initial backfill, cap the
            # historical scan at 3 months rather than a full-history search.
            needs_backfill = rule.initial_backfill_done_at is None
            if needs_backfill:
                backfill_since = datetime.datetime.now(
                    datetime.timezone.utc
                ) - datetime.timedelta(days=INITIAL_BACKFILL_DAYS)
                rule_since_str = backfill_since.strftime("%d-%b-%Y")
                logger.info(
                    "Rule %s (bank=%s, sender=%s) initial backfill — SINCE %s",
                    rule.id,
                    rule.bank,
                    rule.sender,
                    rule_since_str,
                )
            else:
                rule_since_str = since_str

            criteria_parts = []
            if rule.sender:
                criteria_parts.append(f'FROM "{rule.sender}"')
            if rule.subject:
                criteria_parts.append(f'SUBJECT "{rule.subject}"')
            if rule_since_str:
                criteria_parts.append(f"SINCE {rule_since_str}")
            criteria = " ".join(criteria_parts) if criteria_parts else "ALL"

            # stdlib runtime accepts None for the optional CHARSET arg
            # ("no charset"); the type stub narrows it to str. Cast keeps
            # the documented IMAP semantics without sprinkling ignores.
            typ, data = conn.uid("SEARCH", cast(str, None), criteria)
            logger.info(
                "Gmail: SEARCH for rule %s (bank=%s): criteria=%s -> result=%s count=%s",
                rule.id,
                rule.bank,
                criteria,
                typ,
                len(data[0].split()) if typ == "OK" and data[0] else 0,
            )
            if typ != "OK" or not data[0]:
                # SEARCH completed OK but returned zero results — this
                # rule genuinely has no matching emails, so its backfill
                # is complete.
                if needs_backfill and typ == "OK":
                    backfill_searched_rule_ids.add(rule.id)
                continue

            rule_uids = data[0].split()
            for uid in rule_uids:
                uid_to_rules.setdefault((folder, uid), []).append(rule)

        if not uid_to_rules:
            return FetchSourceResult(results_by_rule, True, backfill_searched_rule_ids)

        # Phase 1: batch metadata fetch to get X-GM-MSGID + headers
        # Group UIDs by folder for metadata fetch
        folder_uids: dict[str, list[bytes]] = defaultdict(list)
        for folder, uid in uid_to_rules:
            folder_uids[folder].append(uid)

        uid_meta: dict[tuple, tuple] = {}  # (folder, uid) -> (remote_id, metadata, xgm)
        current_folder = None

        for folder, uids in folder_uids.items():
            if folder != current_folder:
                typ, _ = conn.select(f'"{folder}"', readonly=True)
                if typ != "OK":
                    continue
                current_folder = folder

            # Batch metadata fetch in chunks of 500
            for batch_start in range(0, len(uids), 500):
                batch = uids[batch_start : batch_start + 500]
                # IMAP UIDs are ASCII digits; pass as str per the
                # stdlib stub. (Stdlib runtime accepts bytes too, but
                # we don't need that here.)
                uid_set = b",".join(batch).decode("ascii")
                try:
                    typ, msg_data = conn.uid(
                        "FETCH",
                        uid_set,
                        "(X-GM-MSGID BODY.PEEK[HEADER.FIELDS (MESSAGE-ID FROM SUBJECT DATE)])",
                    )
                except imaplib.IMAP4.error as e:
                    logger.error("Gmail metadata fetch error: %s", e)
                    continue
                if typ != "OK" or not msg_data:
                    continue

                # Parse the batch response — each message is a tuple in msg_data
                for part in msg_data:
                    if isinstance(part, tuple):
                        header_line = part[0]
                        if isinstance(header_line, bytes):
                            # Extract UID from the response line
                            uid_match = re.search(
                                rb"^\*\s+\d+\s+FETCH\s+\(UID\s+(\d+)", header_line
                            )
                            m = re.search(rb"X-GM-MSGID\s+(\d+)", header_line)
                            xgm = m.group(1).decode() if m else None
                            headers_raw = (
                                part[1]
                                if len(part) > 1 and isinstance(part[1], bytes)
                                else None
                            )
                            if headers_raw:
                                msg = email_lib.message_from_bytes(headers_raw)
                                meta = {
                                    "message_id": (msg.get("Message-ID") or "").strip(),
                                    "sender": _decode_header_value(msg.get("From", "")),
                                    "subject": _decode_header_value(
                                        msg.get("Subject", "")
                                    ),
                                    "date": _decode_header_value(msg.get("Date", "")),
                                }
                            else:
                                meta = None

                            # Find which UID this response belongs to
                            resp_uid = None
                            if uid_match:
                                resp_uid = uid_match.group(1)
                            else:
                                # For UID FETCH, the UID is in the response
                                uid_in_resp = re.search(rb"UID\s+(\d+)", header_line)
                                if uid_in_resp:
                                    resp_uid = uid_in_resp.group(1)

                            if resp_uid and meta:
                                remote_id = (
                                    xgm
                                    if xgm
                                    else (
                                        meta["message_id"]
                                        or f"{resp_uid.decode()}:{folder}"
                                    )
                                )
                                uid_meta[(folder, resp_uid)] = (remote_id, meta, xgm)

        # Deduplicate by X-GM-MSGID across rules (same message in multiple folders)
        seen_xgm_set: set[str] = set()
        deduped_keys: list[tuple] = []
        for key in uid_to_rules:
            meta_entry = uid_meta.get(key)
            if not meta_entry:
                # Metadata fetch missed this key; include it to be safe
                deduped_keys.append(key)
                continue
            remote_id, meta, xgm = meta_entry
            if xgm:
                if xgm in seen_xgm_set:
                    continue
                seen_xgm_set.add(xgm)
            deduped_keys.append(key)

        # Phase 1.5: scoped dedup against DB using the precomputed set
        candidate_remote_ids = {}
        for key in deduped_keys:
            meta_entry = uid_meta.get(key)
            if meta_entry:
                candidate_remote_ids[key] = meta_entry[0]

        # Filter to genuinely new keys
        new_keys = []
        for key in deduped_keys:
            remote_id = candidate_remote_ids.get(key)
            if remote_id and remote_id in existing_remote_ids:
                continue
            new_keys.append(key)

        if not new_keys:
            # All UIDs from SEARCH were already in DB — backfill search
            # completed for rules that found UIDs (their data is already
            # persisted from a prior run).
            for key in uid_to_rules:
                for r in uid_to_rules[key]:
                    if r.initial_backfill_done_at is None:
                        backfill_searched_rule_ids.add(r.id)
            return FetchSourceResult(results_by_rule, True, backfill_searched_rule_ids)

        # Apply per-rule fetch limit (scale by number of rules)
        source_limit = fetch_limit * len(rules) if fetch_limit > 0 else 0
        if source_limit > 0:
            new_keys = new_keys[:source_limit]

        # Phase 2: fetch full RFC822 for new emails only
        # Group by folder
        folder_new_keys: dict[str, list[tuple]] = defaultdict(list)
        for key in new_keys:
            folder_new_keys[key[0]].append(key)

        current_folder = None
        for folder, keys in folder_new_keys.items():
            if folder != current_folder:
                typ, _ = conn.select(f'"{folder}"', readonly=True)
                if typ != "OK":
                    continue
                current_folder = folder

            for key in keys:
                uid = key[1]
                try:
                    typ, msg_data = conn.uid("FETCH", uid, "(X-GM-MSGID RFC822)")
                except imaplib.IMAP4.error as e:
                    logger.error("Gmail RFC822 fetch error for uid %s: %s", uid, e)
                    continue
                if typ != "OK" or not msg_data:
                    continue

                xgm_msgid = None
                raw_bytes = None
                for part in msg_data:
                    if isinstance(part, tuple):
                        header_line = part[0]
                        if isinstance(header_line, bytes):
                            m = re.search(rb"X-GM-MSGID\s+(\d+)", header_line)
                            if m:
                                xgm_msgid = m.group(1).decode()
                        raw_bytes = part[1] if len(part) > 1 else raw_bytes

                if raw_bytes is None:
                    continue

                if xgm_msgid:
                    remote_id = xgm_msgid
                else:
                    parsed_msg = email_lib.message_from_bytes(raw_bytes)
                    rfc_msg_id = parsed_msg.get("Message-ID", "").strip()
                    if rfc_msg_id:
                        remote_id = rfc_msg_id
                    else:
                        remote_id = f"{uid.decode()}:{folder}"

                msg_id = f"src{source_id}:gmail:{remote_id}"
                metadata = _extract_message_metadata(raw_bytes)

                # Dispatch to matching rules
                for rule in uid_to_rules.get(key, []):
                    if not _matches_rule_filters(
                        rule,
                        sender=metadata["sender"],
                        subject=metadata["subject"],
                    ):
                        logger.debug(
                            "Skipping Gmail message %s after local rule filter check for rule %s",
                            msg_id,
                            rule.id,
                        )
                        continue
                    results_by_rule[rule.id].append(
                        FetchedEmail(
                            msg_id=msg_id,
                            remote_id=remote_id,
                            raw_bytes=raw_bytes,
                        )
                    )

        # Mark backfill search complete for rules that actually got results
        # through the entire pipeline.  Rules whose SEARCH found UIDs but
        # whose results were dropped (source_limit, filter mismatch, etc.)
        # are intentionally NOT included so their backfill will be retried.
        for rid, results in results_by_rule.items():
            if results:
                backfill_searched_rule_ids.add(rid)

        return FetchSourceResult(results_by_rule, True, backfill_searched_rule_ids)
    except imaplib.IMAP4.error as e:
        logger.error("Gmail IMAP error for source %s: %s", source_id, e)
        return FetchSourceResult(results_by_rule, False, set())
    finally:
        try:
            conn.logout()
        except Exception:
            pass


def _fetch_gmail_single_sync(user: str, password: str, remote_id: str) -> bytes | None:
    """Fetch a single email from Gmail by X-GM-MSGID."""
    try:
        conn = imaplib.IMAP4_SSL("imap.gmail.com")
        conn.login(user, password)
        conn.select('"[Gmail]/All Mail"', readonly=True)
        # Search by X-GM-MSGID. stdlib runtime accepts None for the
        # optional CHARSET arg; cast to satisfy the stricter stub.
        typ, data = conn.uid("SEARCH", cast(str, None), f"X-GM-MSGID {remote_id}")
        if typ != "OK" or not data[0]:
            conn.logout()
            return None
        uid = data[0].split()[-1]
        typ, msg_data = conn.uid("FETCH", uid, "(RFC822)")
        conn.logout()
        if typ != "OK" or not msg_data or not msg_data[0]:
            return None
        return msg_data[0][1]
    except Exception as e:
        logger.error("Failed to fetch Gmail message %s: %s", remote_id, e)
        return None


class GmailProvider:
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
            _fetch_gmail_source_sync,
            rules,
            user=creds.get("user", ""),
            password=creds.get("app_password", ""),
            fetch_limit=fetch_limit,
            source_id=source.id,
            existing_remote_ids=existing_remote_ids,
            last_synced_at=source.last_synced_at,
        )

    async def fetch_single(self, source, remote_id: str) -> bytes | None:
        creds = decrypt_credentials(source.credentials)
        return await asyncio.to_thread(
            _fetch_gmail_single_sync,
            creds.get("user", ""),
            creds.get("app_password", ""),
            remote_id,
        )
