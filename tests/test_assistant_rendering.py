import pytest

from financial_dashboard.services.assistant.rendering import split_plain_text


def test_split_plain_text_reserves_footer_on_every_chunk():
    chunks = split_plain_text("one two three four", footer="Ref: token", limit=18)
    assert len(chunks) > 1
    assert all(len(chunk) <= 18 for chunk in chunks)
    assert all(chunk.endswith("Ref: token") for chunk in chunks)


def test_split_plain_text_rejects_footer_over_limit():
    with pytest.raises(ValueError):
        split_plain_text("hello", footer="x" * 20, limit=10)
