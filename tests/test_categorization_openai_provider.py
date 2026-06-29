"""Tests for the OpenAI-compatible categorization provider and engine dispatch."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from financial_dashboard.services.categorization.gemini import (
    GeminiResult,
    NEEDS_REVIEW,
)

pytestmark = pytest.mark.anyio

_FIELDS = {
    "direction": "debit",
    "amount": "250.00",
    "currency": "INR",
    "channel": "upi",
    "counterparty": "ACME GROCERS",
    "raw_description": "ACME GROCERS ONLINE",
}


def _make_mock_client(content):
    """Build a mock AsyncOpenAI client whose chat.completions.create returns *content*."""
    mock_message = MagicMock()
    mock_message.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_create = AsyncMock(return_value=mock_response)
    mock_completions = MagicMock()
    mock_completions.create = mock_create
    mock_chat = MagicMock()
    mock_chat.completions = mock_completions
    mock_client = MagicMock()
    mock_client.chat = mock_chat
    return mock_client, mock_create


# ---------------------------------------------------------------------------
# openai_provider.classify tests
# ---------------------------------------------------------------------------


async def test_classify_known_slug(monkeypatch):
    from financial_dashboard.services.categorization import openai_provider

    content = json.dumps(
        {"category": "groceries", "confidence": 0.9, "reason": "food store"}
    )
    mock_client, mock_create = _make_mock_client(content)
    monkeypatch.setattr(
        openai_provider,
        "AsyncOpenAI",
        MagicMock(return_value=mock_client),
    )

    result = await openai_provider.classify(
        fields=_FIELDS,
        examples=[],
        active_slugs=["groceries", "dining"],
        api_key="test-key",
        model="gpt-4o-mini",
        base_url="",
    )

    assert result == GeminiResult(slug="groceries", confidence=0.9, reason="food store")
    mock_create.assert_awaited_once()


async def test_classify_unknown_slug_routes_to_needs_review(monkeypatch):
    from financial_dashboard.services.categorization import openai_provider

    content = json.dumps(
        {"category": "unknown_slug_xyz", "confidence": 0.8, "reason": "unclear"}
    )
    mock_client, _ = _make_mock_client(content)
    monkeypatch.setattr(
        openai_provider,
        "AsyncOpenAI",
        MagicMock(return_value=mock_client),
    )

    result = await openai_provider.classify(
        fields=_FIELDS,
        examples=[],
        active_slugs=["groceries", "dining"],
        api_key="test-key",
        model="gpt-4o-mini",
        base_url="",
    )

    assert result.slug == NEEDS_REVIEW
    assert result.confidence == 0.8


async def test_classify_none_content_handled(monkeypatch):
    """A model response with None content should parse as an empty dict → NEEDS_REVIEW."""
    from financial_dashboard.services.categorization import openai_provider

    mock_client, _ = _make_mock_client(None)
    monkeypatch.setattr(
        openai_provider,
        "AsyncOpenAI",
        MagicMock(return_value=mock_client),
    )

    result = await openai_provider.classify(
        fields=_FIELDS,
        examples=[],
        active_slugs=["groceries"],
        api_key="test-key",
        model="gpt-4o-mini",
        base_url="",
    )

    assert result.slug == NEEDS_REVIEW


async def test_classify_base_url_forwarded_to_client(monkeypatch):
    from financial_dashboard.services.categorization import openai_provider

    content = json.dumps({"category": "groceries", "confidence": 0.7, "reason": "r"})
    mock_client, _ = _make_mock_client(content)
    mock_cls = MagicMock(return_value=mock_client)
    monkeypatch.setattr(openai_provider, "AsyncOpenAI", mock_cls)

    await openai_provider.classify(
        fields=_FIELDS,
        examples=[],
        active_slugs=["groceries"],
        api_key="my-key",
        model="gpt-4o-mini",
        base_url="https://proxy.example/v1",
    )

    mock_cls.assert_called_once_with(
        api_key="my-key",
        base_url="https://proxy.example/v1",
        timeout=30.0,
    )


async def test_classify_empty_base_url_passes_none_to_client(monkeypatch):
    from financial_dashboard.services.categorization import openai_provider

    content = json.dumps({"category": "groceries", "confidence": 0.7, "reason": "r"})
    mock_client, _ = _make_mock_client(content)
    mock_cls = MagicMock(return_value=mock_client)
    monkeypatch.setattr(openai_provider, "AsyncOpenAI", mock_cls)

    await openai_provider.classify(
        fields=_FIELDS,
        examples=[],
        active_slugs=["groceries"],
        api_key="my-key",
        model="gpt-4o-mini",
        base_url="",
    )

    mock_cls.assert_called_once_with(api_key="my-key", base_url=None, timeout=30.0)


# ---------------------------------------------------------------------------
# Engine dispatch tests
# ---------------------------------------------------------------------------


async def test_engine_dispatches_to_openai_provider(monkeypatch):
    """When categorization.llm_provider is 'openai', _llm_classify calls openai_provider.classify."""
    from financial_dashboard.services.categorization import engine as eng
    from financial_dashboard.services import settings as svc_settings

    svc_settings._cache["categorization.llm_provider"] = "openai"
    svc_settings._cache["openai.api_key"] = "fake-openai-key"
    svc_settings._cache["openai.model"] = "gpt-4o-mini"
    svc_settings._cache["openai.base_url"] = ""

    openai_called = False

    async def fake_openai_classify(**kwargs):
        nonlocal openai_called
        openai_called = True
        return GeminiResult("groceries", 0.9, "test")

    monkeypatch.setattr(eng.openai_provider, "classify", fake_openai_classify)

    result = await eng._llm_classify(
        fields=_FIELDS,
        examples=[],
        active_slugs=["groceries"],
    )

    assert openai_called
    assert result.slug == "groceries"


async def test_engine_dispatches_to_gemini_by_default(monkeypatch):
    """When categorization.llm_provider is absent/gemini, _llm_classify calls gemini.classify."""
    from financial_dashboard.services.categorization import engine as eng
    from financial_dashboard.services import settings as svc_settings

    # Ensure the cache has no provider override so the default "gemini" path is taken.
    svc_settings._cache.pop("categorization.llm_provider", None)
    svc_settings._cache["gemini.api_key"] = "fake-gemini-key"

    gemini_called = False

    async def fake_gemini_classify(**kwargs):
        nonlocal gemini_called
        gemini_called = True
        return GeminiResult("dining", 0.85, "restaurant")

    monkeypatch.setattr(eng.gemini, "classify", fake_gemini_classify)

    result = await eng._llm_classify(
        fields=_FIELDS,
        examples=[],
        active_slugs=["dining"],
    )

    assert gemini_called
    assert result.slug == "dining"


async def test_engine_dispatches_to_gemini_when_explicitly_set(monkeypatch):
    """When categorization.llm_provider is explicitly 'gemini', gemini.classify is called."""
    from financial_dashboard.services.categorization import engine as eng
    from financial_dashboard.services import settings as svc_settings

    svc_settings._cache["categorization.llm_provider"] = "gemini"
    svc_settings._cache["gemini.api_key"] = "fake-gemini-key"

    gemini_called = False

    async def fake_gemini_classify(**kwargs):
        nonlocal gemini_called
        gemini_called = True
        return GeminiResult("dining", 0.85, "restaurant")

    monkeypatch.setattr(eng.gemini, "classify", fake_gemini_classify)

    result = await eng._llm_classify(
        fields=_FIELDS,
        examples=[],
        active_slugs=["dining"],
    )

    assert gemini_called
    assert result.slug == "dining"
