import asyncio
import datetime as dt
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from financial_dashboard.db.enums import EmailKind
from financial_dashboard.db.models import (
    BalanceSnapshot,
    CasUpload,
    EmailSource,
    FetchRule,
)
from financial_dashboard.integrations.email import orchestrator
from financial_dashboard.services import cas_emails
from financial_dashboard.services import emails as emails_mod
from tests.conftest import new_test_engine

pytestmark = pytest.mark.anyio


async def _source(session, *, label: str = "Gmail Primary"):
    src = EmailSource(
        provider="gmail",
        label=label,
        account_identifier="me@example.com",
        credentials="",
    )
    session.add(src)
    await session.flush()
    return src


def _set_cas_cache(*, enabled: bool, pan: str):
    from financial_dashboard.services import settings as settings_mod

    settings_mod._cache["cas_auto_fetch_enabled"] = "true" if enabled else "false"
    settings_mod._cache["cas_pan"] = pan


async def _auto_rules(session):
    result = await session.execute(
        select(FetchRule).where(FetchRule.auto_managed.is_(True)).order_by(FetchRule.id)
    )
    return result.scalars().all()


async def test_ensure_cas_fetch_rules_tracks_sources_cooldown_and_toggle(session):
    _set_cas_cache(enabled=True, pan="ABCDE1234F")
    src = await _source(session)
    gone = await _source(session, label="Inactive")
    gone_rule = FetchRule(
        provider="gmail",
        source_id=gone.id,
        sender=cas_emails.CAS_SENDERS[0].address,
        bank="cas_nsdl",
        email_kind=EmailKind.CAS_STATEMENT.value,
        enabled=True,
        auto_managed=True,
    )
    session.add(gone_rule)
    gone.active = False
    await session.flush()

    # A second run must not add rules.
    for _ in range(2):
        await cas_emails.ensure_cas_fetch_rules(session)
        await session.flush()

    rules = [rule for rule in await _auto_rules(session) if rule is not gone_rule]
    assert {rule.sender for rule in rules} == {
        s.address for s in cas_emails.CAS_SENDERS
    }
    assert all(rule.enabled and rule.source_id == src.id for rule in rules)
    assert all(rule.email_kind == EmailKind.CAS_STATEMENT.value for rule in rules)
    assert gone_rule.enabled is False

    # SQLite reads the poll stamp back naive. The cooldown must still apply.
    src.cas_last_polled_at = dt.datetime.utcnow() - dt.timedelta(hours=2)
    await session.flush()
    await cas_emails.ensure_cas_fetch_rules(session)
    assert not any(rule.enabled for rule in rules)

    src.cas_last_polled_at = dt.datetime.utcnow() - dt.timedelta(hours=25)
    await cas_emails.ensure_cas_fetch_rules(session)
    assert all(rule.enabled for rule in rules)
    _set_cas_cache(enabled=False, pan="ABCDE1234F")
    await cas_emails.ensure_cas_fetch_rules(session)
    assert not any(rule.enabled for rule in await _auto_rules(session))


async def test_process_cas_email_ingests_or_surfaces_ingest_error(
    session, cas_statement_payload, tmp_path
):
    _set_cas_cache(enabled=True, pan="ABCDE1234F")
    src = await _source(session)
    payload = dict(cas_statement_payload)

    class FakeCasStatement:
        def model_dump(self, mode="json"):
            return payload

    async def _process(log_ref):
        with (
            patch(
                "financial_dashboard.services.statements.cc.extract_pdf_from_email",
                return_value=[("example_cas.pdf", b"%PDF-1.4 fake")],
            ),
            patch(
                "financial_dashboard.integrations.parsers.parse_cas_pdf",
                return_value=FakeCasStatement(),
            ),
            patch("financial_dashboard.services.cas_emails.STATEMENTS_DIR", tmp_path),
        ):
            return await cas_emails.process_cas_email(
                session, b"raw", source_id=src.id, log_ref=log_ref
            )

    payload["summary"] = {**payload["summary"], "grand_total": None}
    result, error = await _process("msg-bad")
    assert result is None
    assert "grand_total" in error
    assert (await session.execute(select(CasUpload))).scalars().all() == []

    payload["summary"] = cas_statement_payload["summary"]
    result, error = await _process("msg-good")
    assert error is None
    upload = await session.get(CasUpload, result["cas_upload_id"])
    assert upload.grand_total == Decimal("200000.00")
    snapshots = (
        (
            await session.execute(
                select(BalanceSnapshot).where(
                    BalanceSnapshot.cas_upload_id == upload.id
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(snapshots) == 1


@pytest.mark.parametrize(
    ("mock_kwargs", "error"),
    [
        ({"return_value": ({"cas_upload_id": 42}, None)}, None),
        ({"return_value": (None, "ingest exploded")}, "ingest exploded"),
        ({"side_effect": RuntimeError("boom")}, "boom"),
    ],
    ids=["success", "error", "raises"],
)
async def test_cas_dispatcher_commits_only_on_success(mock_kwargs, error):
    """A failed CAS ingest must roll back. Else its delete-then-insert drops
    prior rows. A raised error must not crash the poll cycle."""
    process = AsyncMock(**mock_kwargs)
    fake_session = MagicMock()
    fake_session.commit = AsyncMock()
    fake_session.rollback = AsyncMock()
    fake_session.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch.object(emails_mod, "async_session", lambda: fake_session),
        patch("financial_dashboard.services.cas_emails.process_cas_email", process),
    ):
        result = await emails_mod.parse_email_by_kind(
            bank="cas_nsdl",
            email_kind=EmailKind.CAS_STATEMENT.value,
            raw_bytes=b"raw",
            subject="CAS",
            source_id=None,
            log_ref="msg",
        )

    if error is None:
        assert result.error is None
        assert result.stmt_result == {"cas_upload_id": 42}
        fake_session.commit.assert_awaited_once()
        fake_session.rollback.assert_not_awaited()
    else:
        assert error in result.error
        assert result.stmt_result is None
        fake_session.commit.assert_not_awaited()
        fake_session.rollback.assert_awaited_once()


@pytest.mark.parametrize("fetch_ok", [True, False])
async def test_cas_cooldown_stamped_only_on_fetch_success(monkeypatch, fetch_ok):
    """A transient fetch failure must not stamp cas_last_polled_at. Else one
    network blip locks CAS polling for 24h."""
    engine, holder = new_test_engine()
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(orchestrator, "async_session", maker)

    async with maker() as s:
        src = await _source(s)
        s.add(
            FetchRule(
                provider="gmail",
                source_id=src.id,
                sender=cas_emails.CAS_SENDERS[0].address,
                bank="cas_nsdl",
                email_kind=EmailKind.CAS_STATEMENT.value,
                enabled=True,
                auto_managed=True,
            )
        )
        await s.commit()
        source_id = src.id

    _set_cas_cache(enabled=True, pan="ABCDE1234F")
    fake_provider = AsyncMock()
    # Shape: (results_by_rule, fetch_ok, backfill_ready_rule_ids).
    fake_provider.fetch_source.return_value = ({}, fetch_ok, set())

    with patch.object(orchestrator, "get_provider", return_value=fake_provider):
        await orchestrator.poll_all(
            poll_lock=asyncio.Lock(),
            poll_status={
                "state": "idle",
                "started_at": None,
                "finished_at": None,
                "last_stats": None,
                "last_error": None,
                "progress": None,
            },
        )

    async with maker() as s:
        refreshed = await s.get(EmailSource, source_id)
        assert (refreshed.cas_last_polled_at is not None) is fetch_ok
        if not fetch_ok:
            assert refreshed.last_error is not None

    await engine.dispose()
    holder.close()


def test_extract_pdf_handles_text_plain_attachment_with_pdf_filename():
    """CDSL CAS emails (eCAS@cdslstatement.com) attach the PDF as
    Content-Type: text/plain with a `.pdf` filename. Our extractor must
    trust the filename and not just the MIME type — otherwise the CAS
    auto-fetch path drops these emails with `no PDF attachment` even
    though the bytes are a real PDF."""
    import base64
    from email.mime.base import MIMEBase
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    from financial_dashboard.services.statements.cc import extract_pdf_from_email

    fake_pdf = b"%PDF-1.4\nfake content\n%%EOF\n"

    msg = MIMEMultipart()
    msg["Subject"] = "CDSL CAS for APR2026"
    msg["From"] = "eCAS@cdslstatement.com"
    msg.attach(MIMEText("Please find your CAS attached.", "html"))

    # Mimic the CDSL shape: text/plain content-type, attachment disposition,
    # .pdf filename, base64-encoded PDF bytes.
    attachment = MIMEBase("text", "plain")
    attachment.set_payload(base64.b64encode(fake_pdf).decode("ascii"))
    attachment.add_header("Content-Transfer-Encoding", "base64")
    attachment.add_header(
        "Content-Disposition", "attachment", filename="APR2026_AA03378886_TXN.pdf"
    )
    msg.attach(attachment)

    pdfs = extract_pdf_from_email(msg.as_bytes())
    assert len(pdfs) == 1
    assert pdfs[0].filename == "APR2026_AA03378886_TXN.pdf"
    assert pdfs[0].content == fake_pdf
