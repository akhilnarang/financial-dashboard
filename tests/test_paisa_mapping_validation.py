"""Backend-specific Paisa mapping validation at config-save time."""

from copy import deepcopy

import pytest
from cryptography.fernet import Fernet
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import financial_dashboard.config as config_mod
import financial_dashboard.services.settings as settings_mod
from financial_dashboard.db.models import Base
from financial_dashboard.schemas.extensions import PaisaConfigInput
from financial_dashboard.services.paisa import surface

pytestmark = pytest.mark.anyio


@pytest.fixture
async def settings_db(monkeypatch):
    """Isolated settings storage for real save_config round trips."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(settings_mod, "async_session", maker)
    monkeypatch.setattr(
        config_mod.settings, "email_source_master_key", Fernet.generate_key().decode()
    )
    monkeypatch.setattr(config_mod, "_fernet_instance", None)
    yield maker
    await engine.dispose()


def _input(**overrides) -> PaisaConfigInput:
    base = {
        "mode": "connect",
        "base_url": "http://127.0.0.1:7500",
        "ledger_cli": "ledger",
    }
    base.update(overrides)
    return PaisaConfigInput(**base)


@pytest.mark.parametrize(
    ("backend", "account_name", "category_name"),
    [
        ("ledger", "Assets:Bank:Salary Account", "Expenses:Food And Dining"),
        ("hledger", "Assets:Bank:Salary Account", "Expenses:Food And Dining"),
        ("beancount", "Assets:Bank:SalaryAccount", "Expenses:FoodAndDining"),
    ],
)
async def test_mapping_validation_uses_backend_being_saved(
    session, settings_db, backend, account_name, category_name
):
    """Operator overrides save only when legal for the selected renderer."""
    result = await surface.save_config(
        session,
        _input(
            ledger_cli=backend,
            account_mappings={"1": account_name},
            category_mappings={"groceries": category_name},
        ),
    )

    assert result.ok is True
    assert result.config.ledger_cli == backend
    assert result.config.account_mappings == {"1": account_name}
    assert result.config.category_mappings == {"groceries": category_name}


@pytest.mark.parametrize(
    ("mapping_field", "mapping", "error_label", "error_detail"),
    [
        (
            "account_mappings",
            {"1": "Assets:Bank:Salary Account"},
            "Account Mappings",
            "must not contain spaces",
        ),
        (
            "category_mappings",
            {"groceries": "Expenses:Food And Dining"},
            "Category Mappings",
            "must not contain spaces",
        ),
        (
            "account_mappings",
            {"1": "Asset:Bank:SalaryAccount"},
            "Account Mappings",
            "beancount root",
        ),
        (
            "category_mappings",
            {"groceries": "Expenses:food"},
            "Category Mappings",
            "beancount component",
        ),
    ],
)
async def test_beancount_rejects_projection_invalid_operator_overrides(
    session,
    mapping_field,
    mapping,
    error_label,
    error_detail,
):
    result = await surface.save_config(
        session,
        _input(ledger_cli="beancount", **{mapping_field: mapping}),
    )

    assert result.ok is False
    assert any(
        error_label in error and error_detail in error for error in result.errors
    )


async def test_invalid_backend_is_reported_before_mapping_validation(session):
    result = await surface.save_config(
        session,
        _input(
            ledger_cli="quicken",
            account_mappings={"1": "Assets:Bank:Salary Account"},
        ),
    )

    assert result.ok is False
    assert result.errors == [
        "Ledger CLI Backend must be one of: ledger, hledger, beancount."
    ]


async def test_backend_is_canonicalized_before_mapping_validation(session):
    result = await surface.save_config(
        session,
        _input(
            ledger_cli=" BeAnCoUnT ",
            account_mappings={"1": "Assets:Bank:Salary Account"},
        ),
    )

    assert result.ok is False
    assert any(
        "Account Mappings" in error and "beancount" in error for error in result.errors
    )


async def test_mapping_error_does_not_save_any_partial_config(session, settings_db):
    initial = await surface.save_config(
        session,
        _input(
            ledger_cli="ledger",
            auth_username="before",
            account_mappings={"1": "Assets:Bank:Before"},
        ),
    )
    assert initial.ok is True
    before = deepcopy(settings_mod._cache)

    result = await surface.save_config(
        session,
        _input(
            ledger_cli="beancount",
            auth_username="must-not-save",
            account_mappings={"1": "Assets:Bank:Invalid Name"},
        ),
    )

    assert result.ok is False
    assert settings_mod._cache == before
