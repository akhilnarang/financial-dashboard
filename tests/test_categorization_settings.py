import pytest

from financial_dashboard.services import settings as settings_mod
from financial_dashboard.services.settings import (
    SETTINGS_REGISTRY,
    get_grouped_settings,
    get_redact_name_tokens,
    get_self_identifier_tokens,
    get_setting_int,
    parse_form_updates,
)

pytestmark = pytest.mark.anyio


def test_categorization_settings_registered():
    for key in (
        "gemini.api_key",
        "gemini.model",
        "categorization.enabled",
        "categorization.confidence_threshold",
        "categorization.self_identifiers",
        "categorization.hidden_identifiers",
        "category_vocab_version",
    ):
        assert key in SETTINGS_REGISTRY, key
    assert SETTINGS_REGISTRY["gemini.api_key"].secret is True
    # old keys must be gone
    assert "categorization.self_names" not in SETTINGS_REGISTRY
    assert "categorization.redact_names" not in SETTINGS_REGISTRY


async def test_vocab_version_default_reads_as_one():
    assert get_setting_int("category_vocab_version", 1) == 1


def test_get_self_identifier_tokens_returns_only_self(monkeypatch):
    monkeypatch.setitem(
        settings_mod._cache, "categorization.self_identifiers", "alex, 9999"
    )
    monkeypatch.setitem(
        settings_mod._cache, "categorization.hidden_identifiers", "doe, bob"
    )
    tokens = get_self_identifier_tokens()
    assert tokens == ("alex", "9999")


def test_get_redact_name_tokens_returns_union(monkeypatch):
    monkeypatch.setitem(settings_mod._cache, "categorization.self_identifiers", "alex")
    monkeypatch.setitem(
        settings_mod._cache, "categorization.hidden_identifiers", "doe, bob"
    )
    tokens = get_redact_name_tokens()
    assert set(tokens) == {"alex", "doe", "bob"}


def test_get_redact_name_tokens_deduplicates(monkeypatch):
    monkeypatch.setitem(
        settings_mod._cache, "categorization.self_identifiers", "alex, doe"
    )
    monkeypatch.setitem(
        settings_mod._cache, "categorization.hidden_identifiers", "doe, bob"
    )
    tokens = get_redact_name_tokens()
    assert tokens.count("doe") == 1
    assert set(tokens) == {"alex", "doe", "bob"}


def test_vocab_version_is_internal_and_not_form_editable():
    # internal counter: never rendered in the settings UI, never set via the form
    assert SETTINGS_REGISTRY["category_vocab_version"].internal is True
    grouped = get_grouped_settings()
    rendered = {row["key"] for rows in grouped.values() for row in rows}
    assert "category_vocab_version" not in rendered
    # a (stale) form value for it must NOT produce an update that could roll it back
    updates, errors = parse_form_updates({"category_vocab_version": "1"})
    assert "category_vocab_version" not in updates
