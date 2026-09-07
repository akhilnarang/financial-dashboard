import pytest
from sqlalchemy import text

from financial_dashboard.db.models import ExtensionSyncState, Transaction
from financial_dashboard.services.assistant.contracts import ApplyTransactionChanges
from financial_dashboard.services.assistant.mutations import (
    MutationRejected,
    apply_transaction_changes,
)
from financial_dashboard.services.categorization.vocabulary import ensure_category


def _txn():
    return Transaction(bank="test", email_type="test", direction="debit", amount=10)


@pytest.mark.anyio
async def test_patch_omission_and_strict_direction(session):
    txn = _txn()
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
        changes={"category": {"op": "set", "value": "salary"}},
    )
    with pytest.raises(MutationRejected):
        await apply_transaction_changes(session, bad, current_user_message="salary")


@pytest.mark.anyio
async def test_empty_patch_rejected(session):
    txn = _txn()
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes", transaction_id=txn.id, changes={}
    )
    with pytest.raises(MutationRejected):
        await apply_transaction_changes(session, request, current_user_message="why")


@pytest.mark.anyio
async def test_read_style_message_cannot_apply_an_ordinary_mutation(session):
    txn = _txn()
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"category": {"op": "set", "value": "groceries"}},
    )

    for message in (
        "explain the groceries category",
        "Please show the groceries category",
        "I want to understand the groceries category",
    ):
        with pytest.raises(MutationRejected, match="questions cannot change"):
            await apply_transaction_changes(
                session,
                request,
                current_user_message=message,
            )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"category": {"op": "set", "value": "groceries"}},
            "Don't categorize this as groceries",
        ),
        ({"note": {"op": "set", "value": "lunch"}}, "Never set the note to lunch"),
        ({"note": {"op": "clear"}}, "Don't clear the note"),
        ({"category": {"op": "clear"}}, "Never remove the category"),
    ],
)
async def test_scoped_negation_cannot_authorize_patch(session, changes, message):
    await ensure_category(session, "groceries")
    txn = _txn()
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes", transaction_id=txn.id, changes=changes
    )

    with pytest.raises(MutationRejected, match="negated instructions"):
        await apply_transaction_changes(session, request, current_user_message=message)


@pytest.mark.anyio
async def test_negation_does_not_reject_note_payload_or_other_field_clause(session):
    txn = _txn()
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"note": {"op": "set", "value": "Never forget the receipt"}},
    )
    result = await apply_transaction_changes(
        session,
        request,
        current_user_message="Set note to Never forget the receipt, don't change the category",
    )
    assert result.after["note"] == "Never forget the receipt"


@pytest.mark.anyio
async def test_global_no_change_fence_ignores_note_payload(session):
    txn = _txn()
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"note": {"op": "set", "value": "Don't make changes"}},
    )
    result = await apply_transaction_changes(
        session,
        request,
        current_user_message="Set note to Don't make changes",
    )
    assert result.after["note"] == "Don't make changes"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "message",
    [
        "Don't make changes; category groceries",
        "This belongs in groceries, but don't make changes",
        "Don't make changes for now; category groceries",
    ],
)
async def test_global_no_change_fence_rejects_ordinary_mutation(session, message):
    await ensure_category(session, "groceries")
    txn = _txn()
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"category": {"op": "set", "value": "groceries"}},
    )

    with pytest.raises(MutationRejected, match="forbids transaction changes"):
        await apply_transaction_changes(
            session,
            request,
            current_user_message=message,
        )


@pytest.mark.anyio
async def test_global_no_change_fence_rejects_merchant_rule(session):
    await ensure_category(session, "groceries")
    txn = _txn()
    txn.counterparty = "Amazon Fresh"
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"category": {"op": "set", "value": "groceries"}},
        merchant_rule={
            "category": "groceries",
            "intent_evidence": "Always use groceries",
        },
    )

    with pytest.raises(MutationRejected, match="forbids transaction changes"):
        await apply_transaction_changes(
            session,
            request,
            current_user_message="Always use groceries, but don't make changes",
        )


@pytest.mark.anyio
async def test_negation_inside_note_payload_does_not_deny_note_write(session):
    txn = _txn()
    session.add(txn)
    await session.flush()
    value = "Do not forget the note"
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"note": {"op": "set", "value": value}},
    )

    result = await apply_transaction_changes(
        session,
        request,
        current_user_message=f"Set note to {value}",
    )
    assert result.after["note"] == value


@pytest.mark.anyio
async def test_unquoted_note_cannot_absorb_later_clause(session):
    txn = _txn()
    session.add(txn)
    await session.flush()
    value = "Do not forget the note, paid cash"
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"note": {"op": "set", "value": value}},
    )

    with pytest.raises(MutationRejected, match="application-bounded"):
        await apply_transaction_changes(
            session,
            request,
            current_user_message=f"Set note to {value}",
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "message",
    [
        'Set note to "Do not forget the note, paid cash"',
        "Set note to 'Do not forget the note, paid cash'",
    ],
)
async def test_quoted_or_colon_note_payload_preserves_internal_negation(
    session, message
):
    txn = _txn()
    session.add(txn)
    await session.flush()
    value = "Do not forget the note, paid cash"
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"note": {"op": "set", "value": value}},
    )

    result = await apply_transaction_changes(
        session,
        request,
        current_user_message=message,
    )
    assert result.after["note"] == value


@pytest.mark.anyio
async def test_model_selected_note_cannot_remove_global_denial_or_add_category(session):
    await ensure_category(session, "groceries")
    txn = _txn()
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={
            "note": {
                "op": "set",
                "value": "groceries, don't make any changes",
            },
            "category": {"op": "set", "value": "groceries"},
        },
    )

    with pytest.raises(MutationRejected, match="forbids transaction changes"):
        await apply_transaction_changes(
            session,
            request,
            current_user_message="Set note to groceries, don't make any changes",
        )
    assert txn.note is None
    assert txn.category is None


@pytest.mark.anyio
async def test_quoted_note_durable_language_is_only_note_payload(session):
    txn = _txn()
    session.add(txn)
    await session.flush()
    value = "always categorize Swiggy as food"
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"note": {"op": "set", "value": value}},
    )

    result = await apply_transaction_changes(
        session,
        request,
        current_user_message=f'Set note to "{value}"',
    )
    assert result.after["note"] == value


@pytest.mark.anyio
async def test_question_mark_inside_quoted_note_is_only_payload(session):
    txn = _txn()
    session.add(txn)
    await session.flush()
    value = "paid cash? check receipt"
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"note": {"op": "set", "value": value}},
    )

    result = await apply_transaction_changes(
        session,
        request,
        current_user_message=f'Set note to "{value}"',
    )
    assert result.after["note"] == value


@pytest.mark.anyio
async def test_single_quoted_contraction_is_only_note_payload(session):
    txn = _txn()
    session.add(txn)
    await session.flush()
    value = "don't categorize this as groceries"
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"note": {"op": "set", "value": value}},
    )

    result = await apply_transaction_changes(
        session,
        request,
        current_user_message=f"set note to '{value}'",
    )
    assert result.after["note"] == value
    assert result.after["category"] is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    "message",
    [
        "set note to Amazon Fresh and category to groceries",
        "Amazon Fresh, groceries",
        "this was Amazon Fresh, note that and categorize it as groceries",
    ],
)
async def test_combined_note_and_category_instruction_applies_both(session, message):
    await ensure_category(session, "groceries")
    txn = _txn()
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={
            "note": {"op": "set", "value": "Amazon Fresh"},
            "category": {"op": "set", "value": "groceries"},
        },
    )

    result = await apply_transaction_changes(
        session,
        request,
        current_user_message=message,
    )
    assert result.after["note"] == "Amazon Fresh"
    assert result.after["category"] == "groceries"


@pytest.mark.anyio
async def test_combined_note_and_category_rejects_partial_model_patch(session):
    txn = _txn()
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"note": {"op": "set", "value": "Amazon Fresh"}},
    )

    with pytest.raises(MutationRejected, match="requires both"):
        await apply_transaction_changes(
            session,
            request,
            current_user_message=("set note to Amazon Fresh and category to groceries"),
        )


@pytest.mark.anyio
async def test_preservation_request_cannot_become_note_shorthand(session):
    await ensure_category(session, "groceries")
    txn = _txn()
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={
            "note": {"op": "set", "value": "Leave this unchanged"},
            "category": {"op": "set", "value": "groceries"},
        },
    )

    with pytest.raises(MutationRejected, match="cannot change transaction data"):
        await apply_transaction_changes(
            session,
            request,
            current_user_message="Leave this unchanged, groceries",
        )


@pytest.mark.anyio
async def test_modal_uncertainty_cannot_mutate_category(session):
    await ensure_category(session, "salary")
    txn = _txn()
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"category": {"op": "set", "value": "salary"}},
    )

    with pytest.raises(MutationRejected, match="uncertain instructions"):
        await apply_transaction_changes(
            session,
            request,
            current_user_message="This might be salary",
        )


@pytest.mark.anyio
async def test_date_slashes_do_not_make_explicit_category_assignment_ambiguous(
    session,
):
    await ensure_category(session, "groceries")
    txn = _txn()
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"category": {"op": "set", "value": "groceries"}},
    )

    result = await apply_transaction_changes(
        session,
        request,
        current_user_message=(
            "Set category to groceries for the purchase on 07/09/2026"
        ),
    )
    assert result.after["category"] == "groceries"


@pytest.mark.anyio
async def test_modal_denial_cannot_mutate_category(session):
    await ensure_category(session, "groceries")
    txn = _txn()
    txn.category = "dining"
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"category": {"op": "set", "value": "groceries"}},
    )

    with pytest.raises(MutationRejected):
        await apply_transaction_changes(
            session,
            request,
            current_user_message="You might not categorize this as groceries",
        )
    assert txn.category == "dining"


@pytest.mark.anyio
async def test_provider_cannot_hide_denied_note_action_inside_selected_value(session):
    txn = _txn()
    txn.note = "original"
    session.add(txn)
    await session.flush()
    denied = "Never set the note to lunch"
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"note": {"op": "set", "value": denied}},
    )

    with pytest.raises(MutationRejected, match="negated instructions"):
        await apply_transaction_changes(
            session,
            request,
            current_user_message=denied,
        )
    assert txn.note == "original"


@pytest.mark.anyio
async def test_note_value_matching_field_noun_cannot_erase_denied_action(session):
    txn = _txn()
    txn.note = "original"
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"note": {"op": "set", "value": "note"}},
    )

    with pytest.raises(MutationRejected, match="negated instructions"):
        await apply_transaction_changes(
            session,
            request,
            current_user_message="Never set the note to note",
        )
    assert txn.note == "original"


@pytest.mark.anyio
async def test_declining_category_creation_does_not_deny_existing_assignment(session):
    await ensure_category(session, "groceries")
    txn = _txn()
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"category": {"op": "set", "value": "groceries"}},
    )

    result = await apply_transaction_changes(
        session,
        request,
        current_user_message="Set the category to groceries, don't create a category",
    )
    assert result.after["category"] == "groceries"


@pytest.mark.anyio
async def test_note_negation_in_prior_clause_does_not_deny_category_assignment(session):
    await ensure_category(session, "groceries")
    txn = _txn()
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"category": {"op": "set", "value": "groceries"}},
    )

    result = await apply_transaction_changes(
        session,
        request,
        current_user_message="Don't change the note, set the category to groceries",
    )
    assert result.after["category"] == "groceries"


@pytest.mark.anyio
async def test_category_negation_in_prior_clause_does_not_deny_note_write(session):
    txn = _txn()
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"note": {"op": "set", "value": "dinner"}},
    )

    result = await apply_transaction_changes(
        session,
        request,
        current_user_message="Don't change the category, set the note to dinner",
    )
    assert result.after["note"] == "dinner"


@pytest.mark.anyio
async def test_category_negation_on_prior_line_does_not_deny_note_write(session):
    txn = _txn()
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"note": {"op": "set", "value": "dinner"}},
    )

    result = await apply_transaction_changes(
        session,
        request,
        current_user_message="Don't change the category\nSet note to dinner",
    )
    assert result.after["note"] == "dinner"


@pytest.mark.anyio
async def test_polite_cashflow_question_cannot_change_exclusion(session):
    txn = _txn()
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"exclude_from_cashflow": {"op": "set", "value": True}},
    )

    with pytest.raises(MutationRejected, match="questions cannot change"):
        await apply_transaction_changes(
            session,
            request,
            current_user_message="Please explain cashflow exclusion",
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("value", "message"),
    [
        (True, "include this in cashflow"),
        (False, "exclude this from cashflow"),
        (True, "never exclude this from cashflow"),
        (True, "I do not want this excluded from cashflow"),
        (False, "never include this in cashflow"),
    ],
)
async def test_cashflow_intent_must_match_requested_polarity(session, value, message):
    txn = _txn()
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"exclude_from_cashflow": {"op": "set", "value": value}},
    )

    with pytest.raises(MutationRejected, match="current-message intent"):
        await apply_transaction_changes(
            session,
            request,
            current_user_message=message,
        )


@pytest.mark.anyio
async def test_include_cashflow_intent_applies_false_polarity(session):
    txn = _txn()
    txn.exclude_from_cashflow = True
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"exclude_from_cashflow": {"op": "set", "value": False}},
    )

    result = await apply_transaction_changes(
        session,
        request,
        current_user_message="include this in cashflow",
    )

    assert result.after["exclude_from_cashflow"] is False


@pytest.mark.anyio
async def test_rejected_multi_field_patch_rolls_back_earlier_fields(session):
    txn = _txn()
    txn.note = "original"
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={
            "note": {"op": "set", "value": "lunch"},
            "category": {"op": "set", "value": "missing-category"},
        },
    )

    with pytest.raises(MutationRejected):
        await apply_transaction_changes(
            session,
            request,
            current_user_message="set note to lunch and category to missing-category",
        )

    await session.refresh(txn)
    assert txn.note == "original"
    assert txn.category is None


@pytest.mark.anyio
async def test_rejected_mutation_fence_does_not_dirty_paisa_revision(session):
    txn = _txn()
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
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"category": {"op": "set", "value": "missing-category"}},
    )

    with pytest.raises(MutationRejected):
        await apply_transaction_changes(
            session,
            request,
            current_user_message="category missing-category",
        )
    await session.commit()
    await session.refresh(state)

    assert state.desired_revision == 7


@pytest.mark.anyio
async def test_assistant_create_rejects_near_duplicate_category(session):
    await ensure_category(session, "groceries")
    txn = _txn()
    session.add(txn)
    await session.flush()
    request = ApplyTransactionChanges(
        name="apply_transaction_changes",
        transaction_id=txn.id,
        changes={"category": {"op": "set", "value": "grocieis"}},
        create_category={
            "slug": "grocieis",
            "intent_evidence": "create category grocieis",
        },
    )

    with pytest.raises(MutationRejected, match="matches existing"):
        await apply_transaction_changes(
            session,
            request,
            current_user_message="create category grocieis",
        )


@pytest.mark.anyio
async def test_assistant_corrected_category_can_authorize_merchant_rule(session):
    await ensure_category(session, "groceries")
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
