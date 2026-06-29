from financial_dashboard.services.categorization.polarity import resolve_direction


def test_credit_shopping_becomes_repayment():
    slug, changed = resolve_direction("shopping", "credit")
    assert slug == "repayment"
    assert changed is True


def test_credit_entertainment_becomes_repayment():
    slug, changed = resolve_direction("entertainment", "credit")
    assert slug == "repayment"
    assert changed is True


def test_credit_unknown_becomes_repayment():
    slug, changed = resolve_direction("unknown", "credit")
    assert slug == "repayment"
    assert changed is True


def test_debit_refund_becomes_expense():
    slug, changed = resolve_direction("refund", "debit")
    assert slug == "expense"
    assert changed is True


def test_debit_interest_becomes_expense():
    slug, changed = resolve_direction("interest", "debit")
    assert slug == "expense"
    assert changed is True


def test_debit_unknown_becomes_expense():
    slug, changed = resolve_direction("unknown", "debit")
    assert slug == "expense"
    assert changed is True


def test_debit_groceries_unchanged():
    slug, changed = resolve_direction("groceries", "debit")
    assert slug == "groceries"
    assert changed is False


def test_credit_refund_unchanged():
    slug, changed = resolve_direction("refund", "credit")
    assert slug == "refund"
    assert changed is False


def test_credit_self_transfer_unchanged():
    slug, changed = resolve_direction("self_transfer", "credit")
    assert slug == "self_transfer"
    assert changed is False


def test_debit_expense_unchanged():
    slug, changed = resolve_direction("expense", "debit")
    assert slug == "expense"
    assert changed is False


def test_credit_repayment_unchanged():
    slug, changed = resolve_direction("repayment", "credit")
    assert slug == "repayment"
    assert changed is False
