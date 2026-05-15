"""Rule matching helpers."""

from __future__ import annotations

from email import utils as email_utils

from financial_dashboard.db import FetchRule


def _clean_rule_value(value: str | None) -> str:
    return (value or "").strip()


def _normalize_text(value: str | None) -> str:
    return _clean_rule_value(value).lower()


def _sender_matches(rule_sender: str, actual_sender: str) -> bool:
    if not rule_sender:
        return True
    actual = _normalize_text(actual_sender)
    expected = _normalize_text(rule_sender)
    if not actual:
        return False
    addr = email_utils.parseaddr(actual_sender)[1].lower()
    return expected in actual or expected == addr


def _subject_matches(rule_subject: str, actual_subject: str) -> bool:
    if not rule_subject:
        return True
    actual = _normalize_text(actual_subject)
    expected = _normalize_text(rule_subject)
    return bool(actual) and expected in actual


def _matches_rule_filters(
    rule: FetchRule, *, sender: str | None, subject: str | None
) -> bool:
    return _sender_matches(
        _clean_rule_value(rule.sender), sender or ""
    ) and _subject_matches(_clean_rule_value(rule.subject), subject or "")


def _format_jmap_from_field(from_field: list[dict] | None) -> str:
    if not from_field:
        return ""
    parts = []
    for sender in from_field:
        name = _clean_rule_value(sender.get("name"))
        email_addr = _clean_rule_value(sender.get("email"))
        if name and email_addr:
            parts.append(f"{name} <{email_addr}>")
        elif email_addr:
            parts.append(email_addr)
        elif name:
            parts.append(name)
    return ", ".join(parts)
