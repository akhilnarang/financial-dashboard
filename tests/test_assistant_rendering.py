import pytest

from financial_dashboard.services.assistant.rendering import (
    escape_html,
    split_plain_text,
)


def test_escape_html_treats_transaction_fields_as_untrusted():
    assert escape_html('<b onclick="bad">&') == (
        "&lt;b onclick=&quot;bad&quot;&gt;&amp;"
    )


def test_split_plain_text_reserves_footer_on_every_chunk():
    chunks = split_plain_text("one two three four", footer="Ref: token", limit=18)
    assert len(chunks) > 1
    assert all(len(chunk) <= 18 for chunk in chunks)
    assert all(chunk.endswith("Ref: token") for chunk in chunks)


def test_split_plain_text_rejects_footer_over_limit():
    with pytest.raises(ValueError):
        split_plain_text("hello", footer="x" * 20, limit=10)


def test_long_background_review_is_split_with_footer_and_limit():
    chunks = split_plain_text("merchant details " * 1000, footer="Ref: review-token")
    assert len(chunks) > 3
    assert all(len(chunk) <= 4096 for chunk in chunks)
    assert all(chunk.endswith("Ref: review-token") for chunk in chunks)
