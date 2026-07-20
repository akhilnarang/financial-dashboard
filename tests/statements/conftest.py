"""Pytest fixtures scoped to ``tests/statements/``.

Wires an in-memory SQLite session factory into every module that owns a
statement pipeline (``process_*_statement_email``, retry helpers, reminders),
and defaults Telegram notifications, password-hint extraction, and payment
tracking to noops so most tests stay offline and deterministic. Individual
tests override any of these via their own ``monkeypatch``.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from financial_dashboard.db import Base


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def maker(monkeypatch):
    """In-memory SQLite session factory wired into every module that owns a
    ``process_*`` / retry / reminder pipeline via its module-level
    ``async_session`` reference."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    import financial_dashboard.services.reminders as reminders_module
    import financial_dashboard.services.statements.bank as bank_module
    import financial_dashboard.services.statements.cc as cc_module
    import financial_dashboard.services.statements.shared as shared_module

    monkeypatch.setattr(cc_module, "async_session", factory)
    monkeypatch.setattr(bank_module, "async_session", factory)
    monkeypatch.setattr(shared_module, "async_session", factory)
    monkeypatch.setattr(reminders_module, "async_session", factory)
    yield factory
    await engine.dispose()


@pytest.fixture(autouse=True)
def _no_telegram(monkeypatch):
    """Telegram notifications need network + creds; default to off."""
    import financial_dashboard.services.statements.bank as bank_module
    import financial_dashboard.services.statements.cc as cc_module

    monkeypatch.setattr(cc_module, "should_notify_transactions", lambda: False)
    monkeypatch.setattr(bank_module, "should_notify_transactions", lambda: False)


@pytest.fixture(autouse=True)
def _no_password_hint(monkeypatch):
    """Default: no password hint extracted from email bodies. Tests that need
    a hint override ``cc_module.extract_password_hint`` / ``bank_module``."""
    import financial_dashboard.services.statements.bank as bank_module
    import financial_dashboard.services.statements.cc as cc_module

    monkeypatch.setattr(cc_module, "extract_password_hint", lambda *a, **kw: None)
    monkeypatch.setattr(bank_module, "extract_password_hint", lambda *a, **kw: None)


@pytest.fixture(autouse=True)
def _no_payment_tracking(monkeypatch):
    """``init_payment_tracking`` runs its own date-gated logic; default to a
    noop so most tests aren't coupled to ``date.today()``. Tests that need it
    override this fixture. Patches every reference site — cc.py and shared.py
    use function-local imports, while web route modules import it at top
    level."""
    import financial_dashboard.services.reminders as reminders_module
    import financial_dashboard.web.statements as cc_routes

    async def _noop(_upload_id):
        return True

    monkeypatch.setattr(reminders_module, "init_payment_tracking", _noop)
    monkeypatch.setattr(cc_routes, "init_payment_tracking", _noop)


@pytest.fixture
def statements_dir(monkeypatch, tmp_path):
    """Redirect on-disk PDF writes to a tmp dir."""
    import financial_dashboard.services.statements.bank as bank_module
    import financial_dashboard.services.statements.cc as cc_module

    monkeypatch.setattr(cc_module, "STATEMENTS_DIR", tmp_path)
    monkeypatch.setattr(bank_module, "STATEMENTS_DIR", tmp_path)
    return tmp_path
