import datetime
from decimal import Decimal

import pytest

from financial_dashboard.db.models import (
    Account,
    Card,
    Category,
    CategoryReviewDecision,
    Transaction,
)
from financial_dashboard.services import settings as settings_service
from financial_dashboard.services.assistant.transaction_reads import (
    MAX_CONTEXT_ROWS,
    get_transaction,
    list_transactions,
)


def _transaction(**overrides: object) -> Transaction:
    values: dict[str, object] = {
        "bank": "HDFC",
        "email_type": "debit",
        "direction": "debit",
        "amount": Decimal("960.00"),
        "transaction_date": datetime.date(2026, 9, 1),
        "counterparty": "PUREBERRYSMUMBAI",
        "category": "groceries",
        "note": "fruit and snacks",
    }
    values.update(overrides)
    return Transaction(**values)


@pytest.mark.anyio
async def test_get_transaction_redacts_identifiers_carries_review_gate_and_misses_as_none(
    session,
):
    settings_service._cache["categorization.hidden_identifiers"] = "Alice,9000000001"
    transaction = _transaction(
        raw_description="merchant metadata",
        reference_number="123456789",
        note="lunch with Alice",
        review_status="pending",
    )
    session.add(transaction)
    await session.flush()
    session.add(
        CategoryReviewDecision(
            transaction_id=transaction.id,
            category_input_hash="hash",
            candidates_json="[]",
            gate_reason="confidence 0.42 below 0.60",
        )
    )
    await session.flush()

    result = await get_transaction(session, transaction.id)

    assert result is not None
    assert result.id == transaction.id
    assert result.counterparty == "PUREBERRYSMUMBAI"
    # A reference that is not a payment reference is masked as a reference.
    assert result.reference_number == "[redacted-ref]"
    assert result.note == "lunch with [redacted-name]"
    assert result.review_gate_reason == "confidence 0.42 below 0.60"
    assert await get_transaction(session, 404) is None
    # A payment reference names a payment, so the owner sees it whole.
    utr = _transaction(id=transaction.id + 1, reference_number="123456789012")
    session.add(utr)
    await session.flush()
    shown = await get_transaction(session, utr.id)
    assert shown is not None and shown.reference_number == "123456789012"
    # A configured identifier is masked even in a payment-shaped reference.
    phone = _transaction(id=utr.id + 1, reference_number="919000000001")
    session.add(phone)
    await session.flush()
    masked = await get_transaction(session, phone.id)
    assert masked is not None and "9000000001" not in (masked.reference_number or "")


@pytest.mark.anyio
async def test_list_matches_filters_and_exclusion_semantics(session):
    session.add_all(
        [
            _transaction(id=1),
            _transaction(
                id=2,
                account_id=7,
                transaction_date=datetime.date(2026, 8, 1),
                direction="credit",
                category="salary",
                counterparty="ACME",
            ),
            _transaction(id=3, exclude_from_cashflow=True),
            # A bank cut the name short, and the row has no description.
            _transaction(id=4, counterparty="SAMPLE", note=None),
            # A stored text this short is a prefix of too many searches.
            _transaction(id=5, counterparty="SAM", note=None),
            Account(id=17, bank="samplebank", label="Sample Gold+", type="credit_card"),
            Account(id=9, bank="hdfc", label="HDFC Swiggy CC", type="credit_card"),
            Account(id=10, bank="hdfc", label="HDFC Bank", type="bank_account"),
            Card(id=3, account_id=9, card_mask="XXXX0000", label="Pat Holder"),
            Card(id=4, account_id=10, card_mask="XXXX0001", label="HDFC Debit"),
            _transaction(id=10, account_id=10, counterparty="CAFE", note=None),
            Account(
                id=12, bank="hdfc", label="HDFC Swiggy Gold CC", type="credit_card"
            ),
            _transaction(id=12, account_id=12, counterparty="CAFE", note=None),
            Card(id=5, account_id=9, card_mask="XXXX0002", label="Gold Member"),
            _transaction(
                id=13, account_id=9, card_id=5, counterparty="CAFE", note=None
            ),
            Account(id=11, bank="slice", label="slice CC", type="credit_card"),
            _transaction(id=11, account_id=11, counterparty="CAFE", note=None),
            Category(slug="credit_card_payment", active=True),
            _transaction(id=6, bank="samplebank", account_id=17, counterparty="CAFE"),
            _transaction(id=8, account_id=9, counterparty="CAFE", note=None),
            _transaction(
                id=9,
                account_id=9,
                card_id=3,
                counterparty="CAFE",
                category="credit_card_payment",
                note=None,
            ),
        ]
    )
    await session.flush()

    included = await list_transactions(
        session,
        account_id=None,
        date_from=datetime.date(2026, 9, 1),
        direction="debit",
        category="groceries",
        search="berry",
        excluded=False,
    )
    excluded = await list_transactions(session, excluded=True)
    by_full_name = await list_transactions(session, search="Friend Samplename Kumar")
    # The model names an account or a card by its label, in the bank or the
    # search filter.
    by_card_bank = await list_transactions(session, bank="SAMPLE GOLD+ CC")
    by_card_search = await list_transactions(
        session, bank="samplebank", search="Sample Gold+ CC"
    )
    by_part_of_label = await list_transactions(session, bank="Gold+")
    by_holder = await list_transactions(session, bank="Pat's card")
    # A full name does not also pull in an account with a shorter label.
    by_longer_label = await list_transactions(session, bank="HDFC Swiggy CC")
    # The account that the name holds whole beats a card that only holds part
    # of the name.
    by_whole_account = await list_transactions(session, bank="HDFC Swiggy Gold CC")
    # A card and its account in one name mean the card.
    by_card_of_account = await list_transactions(session, bank="Pat's HDFC Swiggy")
    by_card_id = await list_transactions(session, card_id=3)
    # A card named only by its account's words does not narrow the account.
    by_account_with_card = await list_transactions(session, bank="HDFC Bank")
    # A label of one word matches only that exact bank name, never a search,
    # and a name that holds one more word does not match a shorter label.
    by_one_word = await list_transactions(session, bank="slice")
    by_one_word_search = await list_transactions(session, search="slice of cake")
    by_other_product = await list_transactions(session, bank="HDFC Ergo")
    # A merchant word or a person in a search does not pull in the account or
    # the card named after it.
    by_merchant = await list_transactions(session, search="swiggy")
    by_person = await list_transactions(session, search="Pat Holder")
    # The model can name a category by the label that the user sees.
    by_category_label = await list_transactions(session, category="Card bills")

    assert [row.id for row in included] == [1]
    assert [row.id for row in excluded] == [3]
    assert [row.id for row in by_full_name] == [4]
    assert [row.id for row in by_card_bank] == [6]
    assert [row.id for row in by_card_search] == [6]
    assert [row.id for row in by_part_of_label] == [6]
    assert [row.id for row in by_holder] == [9]
    assert [row.id for row in by_longer_label] == [13, 9, 8]
    assert [row.id for row in by_whole_account] == [12]
    assert [row.id for row in by_card_of_account] == [9]
    assert [row.id for row in by_card_id] == [9]
    assert [row.id for row in by_account_with_card] == [10]
    assert [row.id for row in by_one_word] == [11]
    assert by_one_word_search == []
    assert by_other_product == []
    assert by_merchant == []
    assert by_person == []
    assert [row.id for row in by_category_label] == [9]


@pytest.mark.anyio
async def test_list_caps_model_context_rows(session):
    session.add_all(_transaction(id=index) for index in range(1, 25))
    await session.flush()

    result = await list_transactions(session, limit=MAX_CONTEXT_ROWS + 100)

    assert len(result) == MAX_CONTEXT_ROWS
