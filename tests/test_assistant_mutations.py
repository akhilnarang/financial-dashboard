import pytest
from sqlalchemy import text

from financial_dashboard.db.models import (
    Account,
    Category,
    ExtensionSyncState,
    Transaction,
)
from financial_dashboard.services.assistant.contracts import ApplyTransactionChanges
from financial_dashboard.services.assistant.mutations import (
    MutationRejected,
    apply_transaction_changes,
)
from financial_dashboard.services.assistant.intent_policy import (
    derive_merchant_pattern,
    is_direct_affirmative,
    merchant_rule_is_explicit,
)
from financial_dashboard.services.categorization.vocabulary import ensure_category


def _txn():
    return Transaction(bank="test", email_type="test", direction="debit", amount=10)


@pytest.mark.anyio
async def test_patch_omission_and_strict_direction(session):
    account = Account(bank="test", label="Card", type="credit_card")
    session.add_all([account, Category(slug="tax_refund", active=True)])
    await session.flush()
    txn = _txn()
    txn.direction = "credit"
    txn.account_id = account.id
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"note": {"op": "set", "value": "context"}},
    )
    result = await apply_transaction_changes(
        session, request, current_user_message="set note context"
    )
    assert result.after["note"] == "context"
    assert result.after["category"] is None

    bad = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"category": {"op": "set", "value": "tax_refund"}},
    )
    with pytest.raises(MutationRejected, match="transaction direction"):
        await apply_transaction_changes(
            session, bad, current_user_message="set category to tax refund"
        )
    await session.refresh(txn)
    assert txn.category is None
    assert txn.note == "context"


def _request(txn_id, changes, **kwargs):
    return ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn_id,
        changes=changes,
        **kwargs,
    )


def _set(value):
    return {"op": "set", "value": value}


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("message", "changes"),
    [
        ("why", {}),
        ("explain the groceries category", {"category": _set("groceries")}),
        ("Don't categorize this as groceries", {"category": _set("groceries")}),
        ("Don't make changes; category groceries", {"category": _set("groceries")}),
        (
            "Set note to Do not forget the note, paid cash",
            {"note": _set("Do not forget the note, paid cash")},
        ),
        ("wow this was expensive", {"category": _set("expense")}),
        (
            "set note to Amazon Fresh and category to groceries",
            {"note": _set("Amazon Fresh")},
        ),
        (
            "Leave this unchanged, groceries",
            {"note": _set("Leave this unchanged"), "category": _set("groceries")},
        ),
        ("This might be salary", {"category": _set("salary")}),
        ("Never set the note to note", {"note": _set("note")}),
        ("include this in cashflow", {"exclude_from_cashflow": _set(True)}),
        ("exclude this from cashflow", {"exclude_from_cashflow": _set(False)}),
        (
            "the delivery fee was excluded from the bill",
            {"exclude_from_cashflow": _set(True)},
        ),
    ],
)
async def test_unsupported_instruction_leaves_transaction_unchanged(
    session, message, changes
):
    for slug in ("groceries", "expense", "salary"):
        await ensure_category(session, slug)
    txn = _txn()
    txn.note = "original"
    session.add(txn)
    await session.flush()

    with pytest.raises(MutationRejected):
        await apply_transaction_changes(
            session, _request(txn.id, changes), current_user_message=message
        )
    await session.refresh(txn)
    assert txn.note == "original"
    assert txn.category is None
    assert txn.exclude_from_cashflow is False


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("message", "changes"),
    [
        (
            "Set note to Never forget the receipt, don't change the category",
            {"note": _set("Never forget the receipt")},
        ),
        (
            'Set note to "Do not forget the note, paid cash? check receipt"',
            {"note": _set("Do not forget the note, paid cash? check receipt")},
        ),
        (
            "set note to 'don't categorize this as groceries'",
            {"note": _set("don't categorize this as groceries")},
        ),
        (
            "set note to Amazon Fresh and category to groceries",
            {"note": _set("Amazon Fresh"), "category": _set("groceries")},
        ),
        (
            "Amazon Fresh, groceries",
            {"note": _set("Amazon Fresh"), "category": _set("groceries")},
        ),
        (
            "Set category to groceries for the purchase on 07/09/2026",
            {"category": _set("groceries")},
        ),
        (
            "Set the category to groceries, don't create a category",
            {"category": _set("groceries")},
        ),
        (
            "Don't change the note, set the category to groceries",
            {"category": _set("groceries")},
        ),
        ("No, groceries", {"category": _set("groceries")}),
        ("Actually this was groceries, keep the note", {"category": _set("groceries")}),
        (
            "this was Amazon Fresh, note that and categorize it as groceries",
            {"note": _set("Amazon Fresh"), "category": _set("groceries")},
        ),
        ("Set note to Don't make changes", {"note": _set("Don't make changes")}),
        ("include this in cashflow", {"exclude_from_cashflow": _set(False)}),
    ],
)
async def test_supported_instruction_applies_only_named_fields(
    session, message, changes
):
    await ensure_category(session, "groceries")
    txn = _txn()
    txn.note = "original"
    txn.exclude_from_cashflow = True
    session.add(txn)
    await session.flush()

    result = await apply_transaction_changes(
        session, _request(txn.id, changes), current_user_message=message
    )

    expected = {
        "note": "original",
        "category": None,
        "exclude_from_cashflow": True,
    } | {field: patch["value"] for field, patch in changes.items()}
    for field, value in expected.items():
        assert result.after[field] == value


@pytest.mark.anyio
async def test_rejected_multi_field_patch_rolls_back_without_dirtying_paisa(session):
    txn = _txn()
    txn.note = "original"
    state = ExtensionSyncState(
        extension_id="paisa", desired_revision=7, applied_revision=7
    )
    session.add_all([txn, state])
    await session.commit()
    await session.execute(
        text(
            "CREATE TRIGGER test_assistant_paisa_dirty AFTER UPDATE ON transactions "
            "BEGIN UPDATE extension_sync_state SET desired_revision = "
            "desired_revision + 1 WHERE extension_id = 'paisa'; END"
        )
    )
    await session.commit()
    request = _request(
        txn.id, {"note": _set("lunch"), "category": _set("missing-category")}
    )

    with pytest.raises(MutationRejected):
        await apply_transaction_changes(
            session,
            request,
            current_user_message="set note to lunch and category to missing-category",
        )
    await session.commit()
    await session.refresh(txn)
    await session.refresh(state)

    assert txn.note == "original"
    assert txn.category is None
    assert state.desired_revision == 7


@pytest.mark.anyio
async def test_assistant_corrected_category_uses_only_active_category_for_merchant_rule(
    session,
):
    await ensure_category(session, "groceries")
    session.add(Category(slug="grocieis", active=False))
    txn = _txn()
    txn.counterparty = "PUREBERRYSMUMBAI"
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"category": {"op": "set", "value": "grocieis"}},
        merchant_rule={
            "category": "grocieis",
            "intent_evidence": "always make a merchant rule for grocieis",
        },
    )

    result = await apply_transaction_changes(
        session,
        request,
        current_user_message="set category grocieis and always make a merchant rule for grocieis",
    )

    assert result.after["category"] == "groceries"
    assert result.merchant_rule_category == "groceries"


@pytest.mark.anyio
async def test_merchant_rule_only_uses_existing_transaction_category(session):
    await ensure_category(session, "groceries")
    txn = _txn()
    txn.category = "groceries"
    txn.counterparty = "PUREBERRYSMUMBAI"
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={},
        merchant_rule={
            "category": "groceries",
            "intent_evidence": "always make a merchant rule for groceries",
        },
    )

    result = await apply_transaction_changes(
        session,
        request,
        current_user_message="always make a merchant rule for groceries",
    )

    assert result.after["category"] == "groceries"
    assert result.merchant_rule_category == "groceries"


def test_merchant_rule_requires_durable_non_negated_cue():
    assert not merchant_rule_is_explicit(
        "set this to groceries", "set this to groceries", "groceries"
    )
    negated = "I don't always buy from Acme, but put this in dining"
    assert not merchant_rule_is_explicit(negated, negated, "dining")
    assert merchant_rule_is_explicit(
        "Always categorize this as groceries",
        "Always categorize this as groceries",
        "groceries",
    )


def test_pending_confirmation_accepts_only_direct_affirmative():
    assert is_direct_affirmative("go ahead")
    assert not is_direct_affirmative("yes, and change the note")


def test_merchant_pattern_is_normalized_and_rejects_generic_values():
    assert derive_merchant_pattern(" Amazon Fresh ") == "amazon fresh"
    assert derive_merchant_pattern("upi payment to Amazon") == "upi payment to amazon"
    for value in ("upi payment to", "abc 12345678"):
        with pytest.raises(ValueError):
            derive_merchant_pattern(value)
