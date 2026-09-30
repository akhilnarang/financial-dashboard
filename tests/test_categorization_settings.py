from financial_dashboard.services import settings as settings_mod
from financial_dashboard.services.settings import (
    get_grouped_settings,
    get_redact_name_tokens,
    get_self_identifier_tokens,
    parse_form_updates,
)


def test_self_and_redact_name_tokens(monkeypatch):
    monkeypatch.setitem(
        settings_mod._cache, "categorization.self_identifiers", "alex, doe"
    )
    monkeypatch.setitem(
        settings_mod._cache, "categorization.hidden_identifiers", "doe, bob"
    )
    assert get_self_identifier_tokens() == ("alex", "doe")
    tokens = get_redact_name_tokens()
    assert tokens.count("doe") == 1
    assert set(tokens) == {"alex", "doe", "bob"}


def test_vocab_version_is_internal_and_not_form_editable():
    # internal counter: never rendered in the settings UI, never set via the form
    grouped = get_grouped_settings()
    rendered = {row["key"] for rows in grouped.values() for row in rows}
    assert "category_vocab_version" not in rendered
    # a (stale) form value for it must NOT produce an update that could roll it back
    updates, _ = parse_form_updates({"category_vocab_version": "1"})
    assert "category_vocab_version" not in updates


def test_api_keys_are_masked_and_kept_on_blank_post(monkeypatch):
    keys = ("gemini.api_key", "openai.api_key")
    for key in keys:
        monkeypatch.setitem(settings_mod._cache, key, "sk-test-secret")
    rows = {r["key"]: r for rs in get_grouped_settings().values() for r in rs}
    for key in keys:
        assert rows[key]["value"] == ""
        assert rows[key]["is_set"] is True
    # a blank secret field must not clear the stored key
    updates, _ = parse_form_updates({key: "" for key in keys})
    assert not set(keys) & set(updates)


def test_reasoning_effort_accepts_a_valid_level_or_empty_and_rejects_a_typo():
    for value in ("medium", ""):
        updates, errors = parse_form_updates({"openai.reasoning_effort": value})
        assert errors == []
        assert updates["openai.reasoning_effort"] == value
    # A typo would fail every categorization call, so reject it on save.
    updates, errors = parse_form_updates({"openai.reasoning_effort": "meduim"})
    assert errors
    assert "openai.reasoning_effort" not in updates
