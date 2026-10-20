from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Category, Transaction
from financial_dashboard.services.categorization.manual import (
    CategoryDirectionPolicy,
    assign_category_manual,
    assign_category_no_commit,
)
from financial_dashboard.services.categorization.review_io import apply_reviewed_rows
from financial_dashboard.services.categorization.vocabulary import (
    ensure_category,
    get_active_slugs,
)
from financial_dashboard.services.transactions import update_transaction_category

pytestmark = pytest.mark.anyio


async def _txn(session: AsyncSession, **kw) -> Transaction:
    txn = Transaction(
        bank="testbank",
        email_type="x",
        direction="debit",
        amount=Decimal(kw.pop("amount", "5")),
        **kw,
    )
    session.add(txn)
    await session.flush()
    return txn


async def test_update_category_assigns_clears_and_rejects(session: AsyncSession):
    """A manual category resolves a notified review row. Clearing it keeps the
    row resolved. A missing txn returns False (404); an invalid slug raises
    ValueError (400)."""
    await ensure_category(session, "groceries")
    txn = await _txn(session, review_status="notified", review_reason="low confidence")

    assert await update_transaction_category(session, txn.id, "Groceries") == (
        True,
        "groceries",
    )
    assert txn.category == "groceries"
    assert txn.category_method == "manual"
    assert txn.category_confidence == 1.0
    assert txn.review_status == "resolved"
    assert (txn.category_model or "").startswith("manual:")

    assert await update_transaction_category(session, txn.id, "") == (True, None)
    assert txn.category is None
    assert txn.category_method == "manual"
    assert txn.review_status == "resolved"

    assert await update_transaction_category(session, 9999, "Groceries") == (
        False,
        None,
    )
    with pytest.raises(ValueError):
        await update_transaction_category(session, txn.id, "123")


async def test_manual_assign_rejects_unknown_slug(session: AsyncSession):
    """A typo must not mint a junk category."""
    await ensure_category(session, "groceries")
    txn = await _txn(session)

    assert await assign_category_manual(session, txn.id, "grocieis") == (False, None)
    await session.refresh(txn)
    assert txn.category is None
    assert "grocieis" not in await get_active_slugs(session)
    assert await assign_category_manual(session, 9999, "groceries") == (False, None)


async def test_inactive_category_is_manual_only(session: AsyncSession):
    session.add(Category(slug="historical", active=False))
    txn = await _txn(session)

    assert await assign_category_no_commit(
        session,
        txn.id,
        "historical",
        direction_policy=CategoryDirectionPolicy.INFERRED_STRICT,
    ) == (False, None)
    assert txn.category is None

    assert await assign_category_manual(session, txn.id, "historical") == (
        True,
        "historical",
    )
    assert txn.category == "historical"


async def test_apply_reviewed_rows(session: AsyncSession):
    """Blank rows skip. Bad ids, bad slugs, missing rows and identity
    mismatches land in invalid. Valid rows still apply."""
    await ensure_category(session, "groceries")
    good = await _txn(session, amount="100")
    bad_slug = await _txn(session, amount="50")
    mismatch = await _txn(session, amount="300")
    blank = await _txn(session, amount="200")

    def row(txn, category, amount=None):
        return {
            "id": str(txn.id),
            "final_category": category,
            "amount": amount or str(txn.amount),
            "date": "",
            "direction": txn.direction,
        }

    result = await apply_reviewed_rows(
        session,
        [
            row(good, "Groceries"),
            row(blank, ""),
            {"id": "not-an-int", "final_category": "groceries"},
            {"id": "9999", "final_category": "groceries"},
            row(bad_slug, "123"),
            row(mismatch, "groceries", amount="999.99"),
        ],
    )

    assert result.applied == 1
    assert result.skipped == 1
    assert result.invalid == [
        "not-an-int:groceries",
        "9999:missing",
        f"{bad_slug.id}:123",
        f"{mismatch.id}:mismatch",
    ]
    assert good.category == "groceries" and good.category_method == "manual"
    for txn in (bad_slug, mismatch, blank):
        assert txn.category_method != "manual"


async def test_bulk_categorize_and_category_list(client, session: AsyncSession):
    """A dry run and a rejected batch write nothing. One bad slug or id rejects
    every item. A real run applies each field and leaves null fields alone. The
    category list names each slug's bank-scope bucket."""
    for slug in ("groceries", "dining", "credit_card_payment"):
        await ensure_category(session, slug)
    first = await _txn(session)
    second = await _txn(session, category="dining", note="old note")
    await session.commit()
    items = [
        {"id": first.id, "category": "groceries", "exclude_from_cashflow": True},
        {"id": second.id, "note": "new note"},
    ]

    async def _rows():
        await session.refresh(first)
        await session.refresh(second)
        return [
            (t.category, t.category_method, t.note, t.exclude_from_cashflow)
            for t in (first, second)
        ]

    untouched = [(None, None, None, False), ("dining", None, "old note", False)]

    preview = await client.post("/api/transactions/categorize", json={"items": items})
    assert preview.status_code == 200, preview.text
    assert preview.json()["dry_run"] is True
    row = preview.json()["items"][0]
    assert (row["category"], row["category_method"], row["exclude_from_cashflow"]) == (
        "groceries",
        "manual",
        True,
    )
    assert await _rows() == untouched

    bad = await client.post(
        "/api/transactions/categorize",
        json={
            "dry_run": False,
            "items": [
                items[0],
                {"id": second.id, "category": "nope"},
                {"id": 999999, "category": "groceries"},
            ],
        },
    )
    assert bad.status_code == 400
    assert [e["index"] for e in bad.json()["detail"]["errors"]] == [1, 2]
    assert await _rows() == untouched

    applied = await client.post(
        "/api/transactions/categorize", json={"dry_run": False, "items": items}
    )
    assert applied.status_code == 200, applied.text
    assert await _rows() == [
        ("groceries", "manual", None, True),
        ("dining", None, "new note", False),
    ]

    categories = await client.get("/api/categories")
    buckets = {item["slug"]: item["bucket"] for item in categories.json()["items"]}
    assert buckets["credit_card_payment"] == "expense"


async def test_apply_previews_rejects_and_applies_like_single_endpoints(
    client, session: AsyncSession
):
    """A dry run and a rejected batch write nothing. One invalid change rejects
    every change and the 400 names each invalid id. A real run writes every
    change, and a category lands with the same side effects as /category."""
    for slug in ("groceries", "dining"):
        await ensure_category(session, slug)
    first = await _txn(
        session,
        category="dining",
        category_method="llm",
        category_confidence=0.5,
        category_model="llm-x",
        review_status="notified",
    )
    twin = await _txn(
        session,
        category="dining",
        category_method="llm",
        category_confidence=0.5,
        category_model="llm-x",
        review_status="notified",
    )
    second = await _txn(session, note="old note")
    credit = await _txn(session)
    credit.direction = "credit"
    await session.commit()
    first_id, second_id, credit_id, twin_id = first.id, second.id, credit.id, twin.id
    changes = [
        {"id": first_id, "category": "groceries"},
        {"id": second_id, "note": None, "exclude_from_cashflow": True, "voided": True},
        {"id": credit_id, "exclude_from_cashflow": False},
    ]

    async def _rows():
        for txn in (first, second, credit):
            await session.refresh(txn)
        return [
            (t.category, t.category_method, t.note, t.exclude_from_cashflow, t.voided)
            for t in (first, second, credit)
        ]

    untouched = await _rows()
    preview = await client.post(
        "/api/transactions/apply", json={"dry_run": True, "changes": changes}
    )
    assert preview.status_code == 200, preview.text
    version = preview.json()["changed"][0]["changes"]["category_vocab_version"]
    expected = [
        {
            "id": first_id,
            "changes": {
                "category": {"before": "dining", "after": "groceries"},
                "category_method": {"before": "llm", "after": "manual"},
                "category_confidence": {"before": 0.5, "after": 1.0},
                "category_model": {"before": "llm-x", "after": "manual:user"},
                "category_vocab_version": {"before": None, "after": version["after"]},
                "review_status": {"before": "notified", "after": "resolved"},
            },
        },
        {
            "id": second_id,
            "changes": {
                "note": {"before": "old note", "after": None},
                "exclude_from_cashflow": {"before": False, "after": True},
                "voided": {"before": False, "after": True},
            },
        },
    ]
    assert version["after"] is not None
    assert preview.json()["changed"] == expected
    assert preview.json()["unchanged"] == [credit_id]
    assert await _rows() == untouched

    bad = await client.post(
        "/api/transactions/apply",
        json={
            "dry_run": False,
            "changes": [
                changes[0],
                {"id": second_id, "category": "nope"},
                {"id": credit_id, "voided": True},
                {"id": 999999, "note": "x"},
            ],
        },
    )
    assert bad.status_code == 400
    errors = bad.json()["detail"]["errors"]
    assert [e["id"] for e in errors] == [second_id, credit_id, 999999]
    assert await _rows() == untouched
    duplicate = await client.post(
        "/api/transactions/apply", json={"changes": [changes[0], changes[0]]}
    )
    assert duplicate.status_code == 422

    applied = await client.post(
        "/api/transactions/apply", json={"dry_run": False, "changes": changes}
    )
    assert applied.status_code == 200, applied.text
    assert applied.json()["changed"] == expected
    assert await _rows() == [
        ("groceries", "manual", None, False, False),
        (None, None, None, True, True),
        (None, None, None, False, False),
    ]
    again = await client.post(
        "/api/transactions/apply", json={"dry_run": False, "changes": changes}
    )
    assert again.json()["changed"] == []
    assert again.json()["unchanged"] == [first_id, second_id, credit_id]

    single = await client.post(
        f"/api/transactions/{twin_id}/category", json={"category": "groceries"}
    )
    assert single.status_code == 200, single.text
    await session.refresh(twin)

    def _category_state(t: Transaction) -> tuple:
        return (
            t.category,
            t.category_method,
            t.category_confidence,
            t.category_model,
            t.category_vocab_version,
            t.review_status,
        )

    assert _category_state(first) == _category_state(twin)
