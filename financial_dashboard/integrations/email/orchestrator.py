"""Email fetch orchestration."""

from __future__ import annotations

import asyncio
import datetime
import logging
from collections import defaultdict

from sqlalchemy import select

from financial_dashboard.db import Email, EmailSource, FetchRule, async_session
from financial_dashboard.integrations.email.base import get_provider
from financial_dashboard.integrations.email.body import _cleanup_failed_spool
from financial_dashboard.services.emails import _serialize_datetime, handle_polled_email
from financial_dashboard.services.linker import build_link_context
from financial_dashboard.services.settings import (
    get_setting_int,
    should_notify_transactions,
)

logger = logging.getLogger(__name__)


def get_poll_status(poll_status: dict) -> dict:
    status = poll_status
    return {
        "state": status["state"],
        "started_at": _serialize_datetime(status["started_at"]),
        "finished_at": _serialize_datetime(status["finished_at"]),
        "last_stats": status["last_stats"],
        "last_error": status["last_error"],
        "progress": status["progress"],
    }


async def poll_all(*, poll_lock: asyncio.Lock, poll_status: dict) -> dict:
    if poll_lock.locked():
        logger.info(
            "Poll requested while another poll is already running; skipping overlap"
        )
        return {
            "rules": 0,
            "fetched": 0,
            "parsed": 0,
            "failed": 0,
            "skipped": 0,
            "status": "already_running",
        }

    async with poll_lock:
        poll_status["state"] = "polling"
        poll_status["started_at"] = datetime.datetime.now(datetime.UTC)
        poll_status["finished_at"] = None
        poll_status["last_error"] = None
        poll_status["progress"] = {
            "source": "",
            "rule": "",
            "email": "",
            "detail": "Initializing...",
        }

        stats = {"rules": 0, "fetched": 0, "parsed": 0, "failed": 0, "skipped": 0}
        fetch_limit = max(1, get_setting_int("poll_fetch_limit_per_rule", 50))

        try:
            try:
                _cleanup_failed_spool()
            except Exception as exc:
                logger.warning("Failed spool cleanup error: %s", exc)

            # function-local: avoid cycle (services.cas_emails -> db.models -> services)
            from financial_dashboard.services.cas_emails import (
                ensure_cas_fetch_rules,
            )

            async with async_session() as session:
                await ensure_cas_fetch_rules(session)
                await session.commit()

                rules = (
                    (
                        await session.execute(
                            select(FetchRule).where(FetchRule.enabled.is_(True))
                        )
                    )
                    .scalars()
                    .all()
                )
            if not rules:
                logger.info("No enabled fetch rules, nothing to poll")
                poll_status["last_stats"] = stats
                return stats

            async with async_session() as session:
                sources_by_id = {
                    src.id: src
                    for src in (
                        await session.execute(
                            select(EmailSource).where(EmailSource.active.is_(True))
                        )
                    )
                    .scalars()
                    .all()
                }
                link_context = await build_link_context(session)

            inserted_msg_ids: set[str] = set()
            rules_by_source: dict[int, list] = defaultdict(list)
            for rule in rules:
                if rule.source_id and (source := sources_by_id.get(rule.source_id)):
                    rules_by_source[source.id].append(rule)
                elif rule.source_id:
                    logger.warning(
                        "Rule %s references source_id=%s which is missing or inactive, skipping",
                        rule.id,
                        rule.source_id,
                    )
                else:
                    logger.warning(
                        "Rule %s has no source_id (legacy rule with provider=%s), skipping. Please assign an email source to this rule.",
                        rule.id,
                        rule.provider,
                    )

            total_rules = len(rules)
            rule_counter = 0

            for source_id, source_rules in rules_by_source.items():
                source = sources_by_id[source_id]
                source_label = source.label
                for _ in source_rules:
                    stats["rules"] += 1

                poll_status["progress"] = {
                    "source": source_label,
                    "rule": f"{rule_counter + 1}/{total_rules}",
                    "email": "0/?",
                    "detail": f"Fetching from {source_label} ({len(source_rules)} rules)",
                }
                async with async_session() as session:
                    rows = (
                        await session.execute(
                            select(Email.remote_id).where(
                                Email.source_id == source_id,
                                Email.remote_id.is_not(None),
                            )
                        )
                    ).scalars()
                    existing_remote_ids: set[str] = {r for r in rows if r is not None}

                provider = get_provider(source)
                (
                    results_by_rule,
                    fetch_ok,
                    backfill_ready_rule_ids,
                ) = await provider.fetch_source(
                    source,
                    source_rules,
                    fetch_limit=fetch_limit,
                    existing_remote_ids=existing_remote_ids,
                )
                logger.info(
                    "Email fetch completed for source %s, fetched %d total emails",
                    source_id,
                    sum(len(v) for v in results_by_rule.values()),
                )

                for rule in source_rules:
                    rule_counter += 1
                    new_emails = results_by_rule.get(rule.id, [])
                    should_notify = (
                        should_notify_transactions()
                        and rule.initial_backfill_done_at is not None
                    )
                    total_emails = len(new_emails)
                    for email_idx, (msg_id, remote_id, raw_bytes) in enumerate(
                        new_emails, 1
                    ):
                        if msg_id in inserted_msg_ids:
                            continue
                        inserted_msg_ids.add(msg_id)
                        stats["fetched"] += 1
                        poll_status["progress"] = {
                            "source": source_label,
                            "rule": f"{rule_counter}/{total_rules}",
                            "email": f"{email_idx}/{total_emails}",
                            "detail": f"Processing email {email_idx}/{total_emails} from {source_label}",
                        }
                        await handle_polled_email(
                            rule=rule,
                            provider=source.provider,
                            source_id=source_id,
                            msg_id=msg_id,
                            remote_id=remote_id,
                            raw_bytes=raw_bytes,
                            should_notify=should_notify,
                            link_context=link_context,
                            stats=stats,
                        )

                if fetch_ok and backfill_ready_rule_ids:
                    async with async_session() as session:
                        for rule in source_rules:
                            if rule.id not in backfill_ready_rule_ids:
                                continue
                            if rule.initial_backfill_done_at is None:
                                db_rule = await session.get(FetchRule, rule.id)
                                if db_rule:
                                    db_rule.initial_backfill_done_at = (
                                        datetime.datetime.now(datetime.UTC)
                                    )
                        await session.commit()

                async with async_session() as session:
                    src = await session.get(EmailSource, source_id)
                    if src:
                        if fetch_ok:
                            src.last_synced_at = datetime.datetime.now(datetime.UTC)
                            src.last_error = None
                        else:
                            src.last_error = (
                                f"Fetch failed for {source.provider} source {source_id}"
                            )
                        # Attempt-based CAS cooldown stamp: only if the fetch actually
                        # succeeded. A transient IMAP failure shouldn't lock out CAS
                        # polling for 24h.
                        if fetch_ok and any(rule.auto_managed for rule in source_rules):
                            src.cas_last_polled_at = datetime.datetime.now(datetime.UTC)
                        await session.commit()

            logger.info("Poll complete: %s", stats)
            poll_status["last_stats"] = stats
            return stats
        except Exception as exc:
            poll_status["last_error"] = str(exc)
            raise
        finally:
            poll_status["state"] = "idle"
            poll_status["finished_at"] = datetime.datetime.now(datetime.UTC)
            poll_status["progress"] = None
