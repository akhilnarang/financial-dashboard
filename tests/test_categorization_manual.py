from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import (
    Category,
    CategoryReviewDecision,
    Receivable,
    ReceivableLink,
    Transaction,
)
from financial_dashboard.services.assistant.delivery import make_delivery
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
from tests.conftest import card_account

pytestmark = pytest.mark.anyio


async def _txn(session: AsyncSession, **kw) -> Transaction:
    txn = Transaction(
        bank="testbank",
        email_type=kw.pop("email_type", "x"),
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


async def test_apply_previews_rejects_and_applies_like_single_endpoints(
    client, session: AsyncSession
):
    """A dry run is the default and writes nothing. One invalid change rejects
    the batch before any write, and the 400 names each invalid id. A real run
    writes every change with the side effects of /category. The category list
    names each slug's bank-scope bucket."""
    for slug in ("groceries", "dining", "credit_card_payment"):
        await ensure_category(session, slug)
    first = await _txn(
        session,
        category="dining",
        category_method="llm",
        category_confidence=0.5,
        category_model="llm-x",
        review_status="notified",
    )
    second = await _txn(session, note="old note", amount="10.00", currency="USD")
    credit = await _txn(session, email_type="testbank_cc_payment_alert")
    credit.direction = "credit"
    credit.account_id = await card_account(session)
    decision = CategoryReviewDecision(
        transaction_id=first.id, category_input_hash="hash", candidates_json="[]"
    )
    session.add(decision)
    await session.flush()
    delivery = make_delivery(
        recipient_chat_id=7,
        text="choose",
        ordinal=0,
        category_review_decision_id=decision.id,
        transaction_id=first.id,
    )
    session.add(delivery)
    await session.commit()
    first_id, second_id, credit_id = first.id, second.id, credit.id
    changes = [
        {"id": first_id, "category": "groceries"},
        {
            "id": second_id,
            "note": None,
            "exclude_from_cashflow": True,
            "voided": True,
            "amount": "900.00",
            "currency": "INR",
        },
        {"id": credit_id, "exclude_from_cashflow": False},
    ]

    async def _rows():
        for row in (first, second, credit, decision, delivery):
            await session.refresh(row)
        return [
            (t.category, t.category_method, t.note, t.exclude_from_cashflow, t.voided)
            for t in (first, second, credit)
        ] + [decision.status, delivery.status]

    untouched = await _rows()
    preview = await client.post("/api/transactions/apply", json={"changes": changes})
    assert preview.status_code == 200, preview.text
    assert preview.json()["dry_run"] is True
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
                "amount": {"before": "10.00", "after": "900.00"},
                "currency": {"before": "USD", "after": "INR"},
                "amount_source": {"before": "bank", "after": "manual"},
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
    # A card payment, a statement row and a receivable's row keep their amount.
    statement_row = await _txn(
        session, email_type="cc_statement", statement_upload_id=1
    )
    owed = await _txn(session)
    session.add(
        Receivable(
            person="A",
            amount=Decimal("5"),
            links=[ReceivableLink(transaction_id=owed.id, role="charge")],
        )
    )
    statement_id, owed_id = statement_row.id, owed.id
    await session.commit()
    kept = await client.post(
        "/api/transactions/apply",
        json={
            "dry_run": False,
            "changes": [
                {"id": credit_id, "amount": "1"},
                {"id": statement_id, "amount": "1"},
                {"id": owed_id, "currency": "USD"},
            ],
        },
    )
    assert kept.status_code == 400
    assert [e["id"] for e in kept.json()["detail"]["errors"]] == [
        credit_id,
        statement_id,
        owed_id,
    ]

    applied = await client.post(
        "/api/transactions/apply", json={"dry_run": False, "changes": changes}
    )
    assert applied.status_code == 200, applied.text
    assert applied.json()["changed"] == expected
    assert await _rows() == [
        ("groceries", "manual", None, False, False),
        (None, None, None, True, True),
        (None, None, None, False, False),
        "superseded",
        "cancelled",
    ]
    assert (second.amount, second.currency) == (Decimal("900.00"), "INR")
    again = await client.post(
        "/api/transactions/apply", json={"dry_run": False, "changes": changes}
    )
    assert again.json()["changed"] == []
    assert again.json()["unchanged"] == [first_id, second_id, credit_id]

    categories = await client.get("/api/categories")
    buckets = {item["slug"]: item["bucket"] for item in categories.json()["items"]}
    assert buckets["credit_card_payment"] == "expense"
