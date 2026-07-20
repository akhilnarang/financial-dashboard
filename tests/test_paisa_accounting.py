import datetime

import pytest

from financial_dashboard.db.models import Account
from financial_dashboard.services.paisa.accounting import (
    KIND_CONTRA_EXPENSE,
    KIND_EXPENSE,
    KIND_INCOME,
    KIND_INVESTMENT,
    KIND_REPAYMENT,
    KIND_UNKNOWN,
    ProjectionError,
    card_clearing_account,
    category_kind,
    contra_account,
    resolve_account,
)
from financial_dashboard.services.paisa.config import PaisaProjectionConfig


def _config(**overrides) -> PaisaProjectionConfig:
    values = {
        "mode": "project",
        "base_url": "http://127.0.0.1:7500",
        "external_url": "",
        "allow_remote": False,
        "auth_username": "",
        "auth_password": "",
        "generated_path": "",
        "selected_account_ids": (1,),
        "cutover_date": datetime.date(2026, 1, 1),
        "account_mappings": {},
        "category_mappings": {},
        "non_inr_policy": "skip",
        "request_timeout_seconds": 15,
        "ledger_cli": "ledger",
        "fx_rates": {},
    }
    values.update(overrides)
    return PaisaProjectionConfig(**values)


@pytest.mark.parametrize(
    ("backend", "account_type", "expected_kind", "expected_name"),
    [
        ("ledger", "bank_account", "asset", "Assets:Bank:Hdfc:Savings Account"),
        ("hledger", "debit_card", "asset", "Assets:Bank:Hdfc:Savings Account"),
        ("beancount", "bank_account", "asset", "Assets:Bank:Hdfc:SavingsAccount"),
        (
            "ledger",
            "credit_card",
            "liability",
            "Liabilities:Card:Hdfc:Savings Account",
        ),
        (
            "beancount",
            "credit_card",
            "liability",
            "Liabilities:Card:Hdfc:SavingsAccount",
        ),
        ("ledger", "future_type", "asset", "Assets:Bank:Hdfc:Savings Account"),
    ],
)
def test_resolve_account_default_policy_matrix(
    backend, account_type, expected_kind, expected_name
):
    account = Account(
        id=1,
        bank="hdfc",
        label="savings_account",
        type=account_type,
    )

    resolved = resolve_account(account, {}, backend)

    assert resolved.kind == expected_kind
    assert resolved.name == expected_name


@pytest.mark.parametrize(
    ("category", "expected_kind", "ledger_name", "beancount_name"),
    [
        ("salary", KIND_INCOME, "Income:Salary", "Income:Salary"),
        ("refund", KIND_CONTRA_EXPENSE, "Expenses:Refund", "Expenses:Refund"),
        (
            "investment",
            KIND_INVESTMENT,
            "Assets:Investments:Unallocated",
            "Assets:Investments:Unallocated",
        ),
        (
            "repayment",
            KIND_REPAYMENT,
            "Equity:Transfers In",
            "Equity:TransfersIn",
        ),
        ("groceries", KIND_EXPENSE, "Expenses:Groceries", "Expenses:Groceries"),
        (None, KIND_UNKNOWN, "Expenses:Unknown", "Expenses:Unknown"),
    ],
)
def test_category_and_contra_policy_matrix(
    category, expected_kind, ledger_name, beancount_name
):
    slug = (category or "").strip().lower() or "unknown"
    assert category_kind(slug) == expected_kind
    assert contra_account(category, _config(), "ledger") == ledger_name
    assert contra_account(category, _config(), "beancount") == beancount_name


def test_operator_overrides_are_strict_and_card_clearing_uses_same_policy():
    account = Account(id=9, bank="hdfc", label="card", type="credit_card")
    config = _config(
        account_mappings={"9": "Liabilities:Custom Card"},
        category_mappings={"credit_card_payment": "Liabilities:Card Clearing"},
    )

    assert resolve_account(account, config.account_mappings, "ledger").name == (
        "Liabilities:Custom Card"
    )
    assert card_clearing_account(config, "ledger") == "Liabilities:Card Clearing"
    with pytest.raises(ProjectionError, match="invalid ledger name for 'beancount'"):
        resolve_account(account, config.account_mappings, "beancount")
