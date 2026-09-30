"""Tests for the OpenAI-compatible categorization provider."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import omit

from financial_dashboard.services.categorization.llm import (
    NEEDS_REVIEW,
    LlmResult,
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
    mock_client = MagicMock()
    mock_create = AsyncMock(
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content=content))])
    )
    mock_client.chat.completions.create = mock_create
    mock_client.base_url = "https://api.openai.com/v1/"
    mock_client.responses.create = AsyncMock()
    mock_client.with_options.return_value = mock_client
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

    assert result == LlmResult(slug="groceries", confidence=0.9, reason="food store")
    mock_create.assert_awaited_once()
    mock_client.responses.create.assert_not_awaited()

    # A refusal can return None content. It must parse as needs_review.
    mock_create.return_value.choices[0].message.content = None
    result = await openai_provider.classify(
        fields=_FIELDS,
        examples=[],
        active_slugs=["groceries"],
        api_key="test-key",
        model="gpt-4o-mini",
        base_url="",
    )
    assert result.slug == NEEDS_REVIEW


async def test_classify_unknown_slug_never_searches_a_redacted_name(monkeypatch):
    from financial_dashboard.services.categorization import openai_provider

    content = json.dumps(
        {
            "category": "unknown_slug_xyz",
            "confidence": 0.8,
            "reason": "unclear",
            "merchant_lookup": {"name": "Alex Quinn", "city": ""},
        }
    )
    mock_client, _ = _make_mock_client(content)
    monkeypatch.setattr(
        openai_provider,
        "AsyncOpenAI",
        MagicMock(return_value=mock_client),
    )

    result = await openai_provider.classify(
        fields={**_FIELDS, "counterparty": "Alex Quinn"},
        name_tokens=["Alex"],
        examples=[],
        active_slugs=["groceries", "dining"],
        api_key="test-key",
        model="gpt-5.6-luna",
        base_url="",
    )

    assert result.slug == NEEDS_REVIEW
    assert result.confidence == 0.8
    assert result.merchant_search is None
    mock_client.responses.create.assert_not_awaited()


async def test_classify_proxy_base_url_skips_web_search(monkeypatch):
    """A proxy base_url must not run the hosted web_search lookup."""
    from financial_dashboard.services.categorization import openai_provider

    content = json.dumps(
        {
            "category": "groceries",
            "confidence": 0.2,
            "reason": "r",
            "merchant_lookup": {"name": "ACME GROCERS", "city": ""},
        }
    )
    mock_client, _ = _make_mock_client(content)
    mock_cls = MagicMock(return_value=mock_client)
    monkeypatch.setattr(openai_provider, "AsyncOpenAI", mock_cls)

    async def classify(base_url):
        mock_cls.reset_mock()
        await openai_provider.classify(
            fields=_FIELDS,
            examples=[],
            active_slugs=["groceries"],
            api_key="my-key",
            model="gpt-4o-mini",
            base_url=base_url,
        )

    await classify("https://proxy.example/v1")
    mock_cls.assert_called_once_with(
        api_key="my-key", base_url="https://proxy.example/v1", timeout=30.0
    )
    mock_client.responses.create.assert_not_awaited()
    # An empty base_url reaches the client as None.
    await classify("")
    mock_cls.assert_called_once_with(api_key="my-key", base_url=None, timeout=30.0)


async def test_classify_sends_only_a_known_reasoning_effort(monkeypatch):
    """A reasoning model gets the effort level and NO temperature. A typo must
    not reach the API, where it would fail every call."""
    from financial_dashboard.services.categorization import openai_provider

    content = json.dumps({"category": "groceries", "confidence": 0.7, "reason": "r"})
    mock_client, mock_create = _make_mock_client(content)
    monkeypatch.setattr(
        openai_provider, "AsyncOpenAI", MagicMock(return_value=mock_client)
    )

    for effort, sent_effort, sent_temperature in (
        ("medium", "medium", omit),
        ("meduim", omit, 0.0),
    ):
        await openai_provider.classify(
            fields=_FIELDS,
            examples=[],
            active_slugs=["groceries"],
            api_key="my-key",
            model="gpt-5.6-luna",
            base_url="",
            reasoning_effort=effort,
        )
        sent = mock_create.await_args.kwargs
        assert sent["reasoning_effort"] == sent_effort
        assert sent["temperature"] == sent_temperature


async def test_uncertain_merchant_uses_public_search_then_reconsiders(monkeypatch):
    """Inferred numeric/Unicode names work without copying private identifiers into search."""
    from financial_dashboard.services.categorization import openai_provider

    client, classify = _make_mock_client(
        json.dumps(
            {
                "category": "needs_review",
                "confidence": 0.2,
                "reason": "unfamiliar merchant",
                "merchant_lookup": {
                    "name": "Café 24 1234/5678/9012/3456",
                    "city": "Montréal 1234567890",
                },
            }
        )
    )
    second, _ = _make_mock_client(
        json.dumps(
            {
                "category": "dining",
                "confidence": 0.85,
                "reason": "café",
            }
        )
    )
    classify.side_effect = [
        classify.return_value,
        second.chat.completions.create.return_value,
    ]
    source = MagicMock(
        type="url_citation", url="https://example.com/cafe24", title="Café 24"
    )
    client.responses.create.return_value = MagicMock(
        id="resp_search",
        status="completed",
        output_text="Café 24 is a café in Montréal.",
        output=[
            MagicMock(type="web_search_call", status="completed"),
            MagicMock(
                type="message",
                content=[MagicMock(type="output_text", annotations=[source])],
            ),
        ],
    )
    monkeypatch.setattr(openai_provider, "AsyncOpenAI", MagicMock(return_value=client))

    result = await openai_provider.classify(
        fields={
            **_FIELDS,
            "counterparty": "CAFE24",
            "raw_description": "private receipt 1234567890",
        },
        examples=[],
        active_slugs=["groceries", "dining"],
        api_key="secret-key",
        model="gpt-6-luna",
        base_url="",
        reasoning_effort="medium",
    )

    assert result.slug == "dining" and result.confidence == 0.85
    search = client.responses.create.await_args.kwargs
    assert json.loads(search["input"]) == {"merchant": "Café 24", "city": "Montréal"}
    assert all(
        private not in json.dumps(search)
        for private in ("250.00", "receipt", "1234567890", "secret-key")
    )
    assert search["max_tool_calls"] == 1 and search["store"] is False
    assert search["tools"] == [{"type": "web_search", "search_context_size": "low"}]
    client.responses.create.assert_awaited_once()
    assert classify.await_count == 2
    assert "tools" not in classify.await_args.kwargs
    assert "untrusted data" in classify.await_args.kwargs["messages"][0]["content"]
    assert result.merchant_search["sources"] == [
        {"url": source.url, "title": source.title}
    ]
    assert result.merchant_search["initial"]["category"] == "needs_review"


async def test_unsourced_merchant_description_cannot_change_category(monkeypatch):
    from financial_dashboard.services.categorization import openai_provider

    client, classify = _make_mock_client(
        json.dumps(
            {
                "category": "needs_review",
                "confidence": 0.2,
                "reason": "unclear",
                "merchant_lookup": {"name": "ACME GROCERS", "city": ""},
            }
        )
    )
    client.responses.create.return_value = MagicMock(
        id="resp_search",
        status="completed",
        output_text="ACME GROCERS is definitely a store.",
        output=[
            MagicMock(type="web_search_call", status="completed"),
            MagicMock(
                type="message", content=[MagicMock(type="output_text", annotations=[])]
            ),
        ],
    )
    monkeypatch.setattr(openai_provider, "AsyncOpenAI", MagicMock(return_value=client))
    result = await openai_provider.classify(
        fields=_FIELDS,
        examples=[],
        active_slugs=["groceries"],
        api_key="secret-key",
        model="gpt-5.6-luna",
        base_url="",
    )
    assert result.slug == "needs_review" and result.confidence == 0.2
    assert result.merchant_search["status"] == "no_evidence"
    assert result.merchant_search["sources"] == []
    classify.assert_awaited_once()


async def test_engine_dispatches_to_openai_with_configured_settings(monkeypatch):
    """With llm_provider 'openai', the OpenAI provider gets the configured settings."""
    from financial_dashboard.services import settings as svc_settings
    from financial_dashboard.services.categorization import engine as eng

    for key, value in {
        "categorization.llm_provider": "openai",
        "openai.api_key": "fake-openai-key",
        "openai.model": "gpt-5.6-luna",
        "openai.base_url": "",
        "openai.reasoning_effort": "medium",
    }.items():
        monkeypatch.setitem(svc_settings._cache, key, value)

    seen = {}

    async def fake_openai_classify(**kwargs):
        seen.update(kwargs)
        return LlmResult("groceries", 0.9, "test")

    monkeypatch.setattr(eng.openai_provider, "classify", fake_openai_classify)

    result = await eng._llm_classify(
        fields=_FIELDS, examples=[], active_slugs=["groceries"]
    )

    assert result.slug == "groceries"
    assert seen["model"] == "gpt-5.6-luna"
    assert seen["reasoning_effort"] == "medium"
    assert seen["fields"] == _FIELDS
