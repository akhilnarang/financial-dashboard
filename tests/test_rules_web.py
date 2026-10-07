import pytest
from sqlalchemy import select

from financial_dashboard.db.models import FetchRule

pytestmark = pytest.mark.anyio

BASE_FORM = {"bank": "hdfc", "sender": "alerts@example.com", "enabled": "true"}


async def _only_rule(session) -> FetchRule:
    rule = (await session.execute(select(FetchRule))).scalar_one()
    await session.refresh(rule)
    return rule


async def test_create_sets_email_kind_and_pages_show_it(client, session):
    """The kind set on create is stored, listed, and preselected on the edit page."""
    resp = await client.post("/rules", data={**BASE_FORM, "email_kind": "cc_statement"})
    assert resp.status_code == 303
    rule = await _only_rule(session)
    assert rule.email_kind == "cc_statement"

    listing = await client.get("/rules")
    assert listing.status_code == 200
    # Once in the add form, once in the rule row.
    assert listing.text.count("Card statement") == 2

    edit = await client.get(f"/rules/{rule.id}/edit")
    assert edit.status_code == 200
    assert '<option value="cc_statement" selected>' in edit.text


async def test_edit_changes_clears_and_preserves_email_kind(client, session):
    """A kind in the form replaces the value. "auto" clears it. No field keeps it."""
    session.add(FetchRule(provider="gmail", bank="hdfc", email_kind="cc_statement"))
    await session.commit()
    rule = await _only_rule(session)
    url = f"/rules/{rule.id}/edit"

    await client.post(url, data=BASE_FORM)
    assert (await _only_rule(session)).email_kind == "cc_statement"

    await client.post(url, data={**BASE_FORM, "email_kind": "bank_statement"})
    assert (await _only_rule(session)).email_kind == "bank_statement"

    await client.post(url, data={**BASE_FORM, "email_kind": "auto"})
    assert (await _only_rule(session)).email_kind is None


async def test_unknown_email_kind_is_rejected(client, session):
    session.add(FetchRule(provider="gmail", bank="hdfc", email_kind="cc_statement"))
    await session.commit()
    rule = await _only_rule(session)

    edit = await client.post(
        f"/rules/{rule.id}/edit", data={**BASE_FORM, "email_kind": "bogus"}
    )
    create = await client.post("/rules", data={**BASE_FORM, "email_kind": "bogus"})

    assert edit.status_code == 422
    assert create.status_code == 422
    assert (await _only_rule(session)).email_kind == "cc_statement"
