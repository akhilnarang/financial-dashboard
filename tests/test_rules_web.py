import re

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import FetchRule

pytestmark = pytest.mark.anyio

BASE_FORM = {"bank": "hdfc", "sender": "alerts@example.com", "enabled": "true"}


async def _only_rule(session: AsyncSession) -> FetchRule:
    rule = (await session.execute(select(FetchRule))).scalar_one()
    await session.refresh(rule)
    return rule


async def test_create_sets_email_kind_and_pages_show_it(
    client: AsyncClient, session: AsyncSession
) -> None:
    """The kind set on create is stored, listed, and preselected on the edit page."""
    resp = await client.post("/rules", data={**BASE_FORM, "email_kind": "cc_statement"})
    assert resp.status_code == 303
    rule = await _only_rule(session)
    assert rule.email_kind == "cc_statement"

    listing = await client.get("/rules")
    assert listing.status_code == 200
    rule_rows = listing.text.split("<tbody>")[1]
    assert "Card statement" in rule_rows

    edit = await client.get(f"/rules/{rule.id}/edit")
    assert edit.status_code == 200
    kind_select = edit.text.split('name="email_kind"')[1].split("</select>")[0]
    selected = re.search(r'value="(\w+)" selected', kind_select)
    assert selected is not None and selected[1] == "cc_statement"


async def test_edit_changes_clears_and_preserves_email_kind(
    client: AsyncClient, session: AsyncSession
) -> None:
    """A kind in the form replaces the value. "auto" clears it. No field keeps it."""
    session.add(FetchRule(provider="gmail", bank="hdfc", email_kind="cc_statement"))
    await session.commit()
    rule = await _only_rule(session)
    url = f"/rules/{rule.id}/edit"

    for missing in ({}, {"email_kind": ""}):
        sender = f"alerts{len(missing)}@example.com"
        form = {**BASE_FORM, "sender": sender, **missing}
        assert (await client.post(url, data=form)).status_code == 303
        rule = await _only_rule(session)
        assert (rule.sender, rule.email_kind) == (sender, "cc_statement")

    await client.post(url, data={**BASE_FORM, "email_kind": "bank_statement"})
    assert (await _only_rule(session)).email_kind == "bank_statement"

    await client.post(url, data={**BASE_FORM, "email_kind": "auto"})
    assert (await _only_rule(session)).email_kind is None


async def test_unknown_email_kind_is_rejected(
    client: AsyncClient, session: AsyncSession
) -> None:
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
