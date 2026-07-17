"""Two-lane synthetic DB loader.

Lane 1 — **fidelity**: a representative subset flows through the dashboard's
real repository services so a regression in them surfaces here. It exercises
**exactly** these production code paths (no others):

* :func:`financial_dashboard.services.cas_ingestion.ingest_cas_payload` — the
  CAS NSDL/CDSL canonicalization + idempotent upsert (incl. its
  ``portfolio_key``/``statement_date`` unique index).
* :func:`financial_dashboard.services.snapshots.upsert_manual_snapshot`,
  :func:`~financial_dashboard.services.snapshots.emit_cc_snapshot` and
  :func:`~financial_dashboard.services.snapshots.emit_bank_snapshot` — the
  balance-snapshot upserts behind manual items and parsed statements.
* :func:`financial_dashboard.services.linker.build_link_context` /
  :func:`~financial_dashboard.services.linker.link_transaction` — account/card
  resolution by mask + number.
* :func:`financial_dashboard.services.txn_merge.merge_transaction` — the
  natural-key (``bank``, ``reference_number``, ``direction``) idempotent
  create/enrich path, including the cross-channel email↔SMS merge.

It deliberately does **not** exercise: the external e-mail/SMS parsers
(``integrations.parsers`` / ``integrations.email``) — rows are synthesised
directly rather than parsed out of synthetic HTML/SMS; the categorizer
(``services.categorization``) — categories are seeded onto each row; the
net-worth pipeline, fetch/polling, and Telegram services.

Lane 2 — **bulk**: the high-volume remainder is inserted via chunked
SQLAlchemy Core ``executemany`` into a dedicated synthetic SQLite file with
``ON CONFLICT DO NOTHING``. The chunk size is capped by SQLite's per-statement
variable limit (see :func:`_chunked_insert`), so a stress profile can land
200k+ rows without exceeding the 999-bind ceiling. This is the lane that lets
a stress profile land 200k+ rows in a few seconds.

Safety
------
The loader only ever writes to a path that passes
:func:`safety.assert_synthetic_db_path` (under ``data/synthetic``). The
default path never names or touches the production DB. Reruns with the same
``(seed, as_of, profile)`` add zero rows: structural rows and bulk rows use
explicit primary keys with ``ON CONFLICT DO NOTHING``; fidelity emails/SMS
are gated on their natural keys so their ``merge_transaction`` calls never
fire twice.

Idempotency
-----------
* structural + bulk rows: explicit PKs + ``on_conflict_do_nothing``
* fidelity emails: skipped when ``message_id`` already exists
* fidelity SMS: ``on_conflict_do_nothing`` on ``(sender, received_at, body)``
* fidelity transactions: created only when their source email did not exist

Network
-------
The loader is **offline by design**. It never opens a socket: emails/SMS are
synthesized directly into the DB (not fetched), CAS payloads are pre-built
dicts (not downloaded), and Paisa is never spawned. The orphan emails, balance
snapshots and statement links added in the realism pass are likewise local DB
writes. If a future extension needs the network (it should not), it must be
opt-in and disabled by default here — the synthetic seed must remain safe to
run with no connectivity and no credentials.
"""

import datetime
import logging
from decimal import Decimal
from pathlib import Path

from sqlalchemy import bindparam, select, text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from scripts.synth import constants as C
from scripts.synth.identity import (
    GENERATOR_VERSION_SETTING_KEY,
    IDENTITY_SETTING_KEY,
    identity_matches,
    load_identity,
)
from scripts.synth.ids import bulk_txn_pk
from scripts.synth.models import Scenario
from scripts.synth.safety import assert_synthetic_db_path
from scripts.synth.scenario import PROFILES

logger = logging.getLogger("scripts.synth.loader")

# Importing production models runs ``financial_dashboard.db.__init__``, which
# builds the *default* engine but does not connect it. We never reference that
# engine; we create our own pointing at the synthetic file.
from financial_dashboard.db.models import (  # noqa: E402
    Account,
    BankStatementUpload,
    Base,
    Card,
    CasUpload,
    Category,
    Email,
    EmailSource,
    FetchRule,
    ManualItem,
    MerchantRule,
    Setting,
    SmsMessage,
    StatementUpload,
    Transaction,
)

CHUNK_SIZE = 5000


class LoaderStats(dict):
    """Per-table insert counts for one load run (str keys, int values)."""


def _db_url(db_path: Path) -> str:
    return f"sqlite+aiosqlite:///{db_path}"


async def create_synthetic_engine(db_path: str | Path):
    """Create an async engine for the (already-validated) synthetic DB path."""
    path = assert_synthetic_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_async_engine(_db_url(path), echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine


async def drop_synthetic_db(db_path: str | Path) -> None:
    """Delete the synthetic DB file (and SQLite sidecars). Path must pass the
    synthetic guard. Caller must have already confirmed via ``safety``."""
    path = assert_synthetic_db_path(db_path)
    for suffix in ("", "-wal", "-shm", "-journal"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()


async def load_scenario(
    scenario: Scenario,
    db_path: str | Path,
    *,
    fidelity_txn_count: int | None = None,
) -> LoaderStats:
    """Load ``scenario`` into the synthetic DB at ``db_path``.

    ``fidelity_txn_count`` overrides the profile default for how many
    transactions flow through the real-service fidelity lane; the rest go
    through the bulk lane. Returns a per-table insert-count dict (the rows
    actually inserted this run — zero on a same-shape rerun).

    Safe over an existing DB for two cases:

    * **same-shape rerun**: the stored identity matches, so no reset happens
      and the natural-key idempotency (explicit PKs + ``ON CONFLICT DO
      NOTHING`` + message_id gating) makes the rerun a no-op (stats all 0).
    * **shape upgrade** (new generator version / seed / profile / code): the
      stored identity differs (or is missing), so the loader wipes every
      loader-owned table in one transaction and rebuilds from scratch —
      guaranteeing the resulting DB is exactly the current scenario with no
      stale rows and no PK collisions. The identity stamp is written *only* on
      full success, so a failed/partial load leaves a stale/missing stamp and
      the next run resets and rebuilds (recoverable).
    """
    path = assert_synthetic_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_async_engine(_db_url(path), echo=False)
    stats = LoaderStats()
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        sessionmaker = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

        if fidelity_txn_count is None:
            fidelity_txn_count = PROFILES[scenario.profile].fidelity_txns

        # Detect a shape mismatch against the previously-loaded corpus. On
        # mismatch, wipe every loader-owned table in one transaction so the
        # upgrade is clean (no PK collisions, no stale rows). On a match, skip
        # the reset and rely on the natural-key idempotency (rerun = no-op).
        async with sessionmaker() as session:
            if await _needs_reset(session, scenario):
                logger.info(
                    "scenario shape changed (seed=%s profile=%s); resetting "
                    "synthetic corpus before load",
                    scenario.seed,
                    scenario.profile,
                )
                await _reset_scenario_tables(session)
                await session.commit()
            else:
                await session.rollback()

        # Lane 1: structural + fidelity (real services).
        async with sessionmaker() as session:
            await _load_structural(session, scenario, stats)
            await _load_fidelity_transactions(
                session, scenario, fidelity_txn_count, stats
            )
            await session.commit()

        # Lane 2: bulk (chunked Core inserts) for the remainder.
        async with sessionmaker() as session:
            await _load_bulk_transactions(session, scenario, fidelity_txn_count, stats)
            await session.commit()

        # Stamp the identity only after both lanes committed (full success), so
        # a failed/partial load does not record a matching stamp and the next
        # run resets + rebuilds.
        async with sessionmaker() as session:
            await _stamp_identity(session, scenario)
            await session.commit()
    finally:
        await engine.dispose()
    return stats


# ---------------------------------------------------------------------------
# Shape-mismatch reset: safe reload over an existing synthetic DB
# ---------------------------------------------------------------------------


#: Loader-owned tables wiped on a shape mismatch, in an order that respects
#: foreign keys when ``PRAGMA foreign_keys`` happens to be on. The
#: ``sms_messages`` ↔ ``transactions`` cycle is harmless because every table is
#: cleared. ``settings`` is included so the stale identity stamp is wiped too;
#: the load re-stamps it on full success.
_RESET_TABLES: tuple[str, ...] = (
    "snapshot_holdings",
    "balance_snapshots",
    "investment_lots",
    "transactions",
    "sms_messages",
    "emails",
    "statement_uploads",
    "bank_statement_uploads",
    "cas_uploads",
    "manual_items",
    "fetch_rules",
    "merchant_rules",
    "cards",
    "accounts",
    "categories",
    "email_sources",
    "extension_runs",
    "settings",
)


async def _read_identity(session: AsyncSession) -> str | None:
    """The last successfully-loaded scenario identity, or None if unstamped."""
    row = (
        await session.execute(
            select(Setting.value).where(Setting.key == IDENTITY_SETTING_KEY)
        )
    ).scalar_one_or_none()
    return str(row) if row is not None else None


async def _needs_reset(session: AsyncSession, scenario: Scenario) -> bool:
    """True when the DB's stamped identity does not match the current scenario.

    A missing stamp (fresh DB, old pre-stamp DB, or a partially-loaded DB whose
    stamp was never written) is always a mismatch, so the next load resets and
    rebuilds — the recovery guarantee for a failed/partial prior load."""
    return not identity_matches(await _read_identity(session), scenario)


async def _reset_scenario_tables(session: AsyncSession) -> None:
    """Wipe every loader-owned table in the current transaction.

    Foreign keys are disabled first (the ``sms_messages`` ↔ ``transactions``
    cycle would otherwise block a plain ``DELETE``), then every loader-owned
    table is cleared and the SQLite autoincrement sequence is reset so a
    rebuilt fidelity lane (which takes autoincrement ids from 1) matches a
    fresh load exactly. Never touches a path outside the synthetic DB the
    session is already bound to (the path guard ran in :func:`load_scenario`)."""
    await session.execute(text("PRAGMA foreign_keys = OFF"))
    for tbl in _RESET_TABLES:
        await session.execute(text(f"DELETE FROM {tbl}"))
    # Reset the SQLite autoincrement sequence so a rebuilt fidelity lane (whose
    # ids are assigned by ``merge_transaction`` starting at 1) matches a fresh
    # load exactly. ``sqlite_sequence`` only exists when a table uses the
    # ``AUTOINCREMENT`` keyword; the dashboard's tables use implicit rowid
    # autoincrement, so the table is usually absent and this is a no-op.
    has_seq = (
        await session.execute(
            text(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'"
            )
        )
    ).scalar_one_or_none()
    if has_seq is not None:
        await session.execute(
            text("DELETE FROM sqlite_sequence WHERE name IN :names").bindparams(
                bindparam("names", expanding=True)
            ),
            {"names": list(_RESET_TABLES)},
        )


async def _stamp_identity(session: AsyncSession, scenario: Scenario) -> None:
    """Record the current scenario's identity + version on full load success.

    Uses the same upsert path as the structural settings, so it is idempotent
    on a same-shape rerun."""
    await _upsert(
        session,
        Setting,
        key=IDENTITY_SETTING_KEY,
        value=load_identity(scenario),
        _pk_name="key",
    )
    await _upsert(
        session,
        Setting,
        key=GENERATOR_VERSION_SETTING_KEY,
        value=C.GENERATOR_VERSION,
        _pk_name="key",
    )


# ---------------------------------------------------------------------------
# Lane 1: structural entities (always) + fidelity transactions
# ---------------------------------------------------------------------------


async def _load_structural(
    session: AsyncSession, scenario: Scenario, stats: LoaderStats
) -> None:
    # All SMS rows are loaded up front (there are few — one per paired event).
    # The fidelity lane then looks them up by id for the cross-channel merge;
    # loading them here keeps the loaded SMS count equal to the scenario's
    # regardless of the fidelity/bulk split.
    for sms in scenario.sms:
        await _upsert(
            session,
            SmsMessage,
            id=sms.pk,
            bank=sms.bank,
            sender=sms.sender,
            body=sms.body,
            received_at=sms.received_at,
            status=sms.status,
        )

    # Accounts / cards / categories / sources / rules / merchant rules.
    for acct in scenario.accounts:
        await _upsert(
            session,
            Account,
            id=acct.pk,
            bank=acct.bank,
            label=acct.label,
            type=acct.type,
            account_number=acct.account_number,
            statement_password=acct.statement_password,
            statement_password_hint=acct.statement_password_hint,
            active=acct.active,
        )
    stats["accounts"] = len(scenario.accounts)
    for card in scenario.cards:
        await _upsert(
            session,
            Card,
            id=card.pk,
            account_id=card.account_pk,
            card_mask=card.card_mask,
            label=card.label,
            is_primary=card.is_primary,
            active=card.active,
        )
    stats["cards"] = len(scenario.cards)
    for cat in scenario.categories:
        await _upsert(
            session, Category, slug=cat.slug, active=cat.active, _pk_name="slug"
        )
    stats["categories"] = len(scenario.categories)
    for src in scenario.email_sources:
        await _upsert(
            session,
            EmailSource,
            id=src.pk,
            provider=src.provider,
            label=src.label,
            account_identifier=src.account_identifier,
            credentials="synthetic",
            active=src.active,
        )
    stats["email_sources"] = len(scenario.email_sources)
    for rule in scenario.fetch_rules:
        await _upsert(
            session,
            FetchRule,
            id=rule.pk,
            provider=rule.provider,
            source_id=rule.source_pk,
            sender=rule.sender,
            subject=rule.subject,
            bank=rule.bank,
            email_kind=rule.email_kind,
            enabled=rule.enabled,
            auto_managed=False,
        )
    stats["fetch_rules"] = len(scenario.fetch_rules)
    # One seed merchant rule so the rules table is exercised.
    await _upsert(
        session,
        MerchantRule,
        pattern="BIGBASKET",
        category="groceries",
        active=True,
        priority=100,
        _pk_name="pattern",
    )
    # A single synthetic setting marks the DB as ours.
    await _upsert(
        session,
        Setting,
        key="synthetic.loaded",
        value=scenario.profile,
        _pk_name="key",
    )

    # Manual items + snapshots via the real service.
    for item in scenario.manual_items:
        existing = await session.get(ManualItem, item.pk)
        if existing is not None:
            continue
        # create_item assigns its own id; pre-set by re-hydration is awkward, so
        # insert directly to keep the deterministic PK.
        session.add(
            ManualItem(
                id=item.pk,
                name=item.name,
                kind=item.kind,
                category=item.category,
                active=item.active,
                notes=item.notes,
            )
        )
        await session.flush()
        if item.active:
            from financial_dashboard.db.enums import (
                ManualKind,
                SnapshotCategory,
                SnapshotKind,
                SnapshotSource,
            )
            from financial_dashboard.services.snapshots import upsert_manual_snapshot

            snap_cat = (
                SnapshotCategory.manual_asset.value
                if item.kind == ManualKind.asset.value
                else SnapshotCategory.manual_liability.value
            )
            await upsert_manual_snapshot(
                session,
                manual_item_id=item.pk,
                kind=SnapshotKind(item.kind),
                category=snap_cat,
                as_of_date=item.as_of_date,
                value=item.value,
                source=SnapshotSource.manual,
            )
    stats["manual_items"] = sum(1 for i in scenario.manual_items if i.active)

    # CAS uploads via the real ingestion service.
    from financial_dashboard.services.cas_ingestion import ingest_cas_payload

    for cas in scenario.cas_uploads:
        # The service upserts on (portfolio_key, statement_date); a rerun is a
        # no-op data-wise but still returns a row. Guard explicitly to keep the
        # reported insert count honest.
        existing = await session.get(CasUpload, cas.pk)
        if existing is not None:
            continue
        # Ingest assigns its own id; we accept that CAS ids are not the
        # scenario's deterministic pk (they remain stable on rerun thanks to
        # the unique index).
        await ingest_cas_payload(session, cas.raw_payload)
    stats["cas_uploads"] = len(scenario.cas_uploads)

    # Statement uploads + their snapshots (bank/cc) via real emitters.
    from financial_dashboard.services.snapshots import (
        emit_bank_snapshot,
        emit_cc_snapshot,
    )

    for stmt in scenario.statement_uploads:
        existing = await session.get(
            StatementUpload if stmt.card_number else BankStatementUpload, stmt.pk
        )
        if existing is not None:
            continue
        if stmt.card_number:
            upload = StatementUpload(
                id=stmt.pk,
                account_id=stmt.account_pk,
                email_id=stmt.email_pk,
                bank=stmt.bank,
                filename=stmt.filename,
                file_path=stmt.file_path,
                source_kind=stmt.source_kind,
                status=stmt.status,
                card_number=stmt.card_number,
                statement_name=stmt.statement_name,
                due_date=stmt.due_date,
                total_amount_due=stmt.total_amount_due,
                minimum_amount_due=stmt.minimum_amount_due,
                payment_status=stmt.payment_status,
                payment_sent_offsets="[]",
                # Truthful reconciliation counts: the transactions the scenario
                # actually linked to this statement (statement_upload_id).
                parsed_txn_count=stmt.parsed_txn_count,
                matched_count=stmt.matched_count,
                imported_count=stmt.imported_count,
            )
            session.add(upload)
            await session.flush()
            await emit_cc_snapshot(session, upload)
        else:
            upload = BankStatementUpload(
                id=stmt.pk,
                account_id=stmt.account_pk,
                email_id=stmt.email_pk,
                bank=stmt.bank,
                filename=stmt.filename,
                file_path=stmt.file_path,
                status=stmt.status,
                closing_balance=stmt.closing_balance,
                statement_period_end=stmt.statement_period_end,
                parsed_txn_count=stmt.parsed_txn_count,
                matched_count=stmt.matched_count,
                imported_count=stmt.imported_count,
            )
            session.add(upload)
            await session.flush()
            await emit_bank_snapshot(session, upload)
    stats["statement_uploads"] = len(scenario.statement_uploads)

    # Standalone pending/failed/skipped emails (not tied to a transaction).
    # Loaded structurally by message_id (natural key); their pks (9001+) never
    # collide with the transaction-email range the lanes create.
    for mail in scenario.orphan_emails:
        await _upsert(
            session,
            Email,
            id=mail.pk,
            provider=mail.provider,
            message_id=mail.message_id,
            source_id=mail.source_pk,
            sender=mail.sender,
            subject=mail.subject,
            received_at=mail.received_at,
            status=mail.status,
            error=mail.error,
        )

    # Balance-derived opening/current snapshots for every active account, so the
    # net-worth surface has truthful figures rather than one manual patch. Uses
    # the real snapshot upsert (asset for bank, liability for card).
    from financial_dashboard.db.enums import (
        SnapshotCategory,
        SnapshotKind,
        SnapshotSource,
    )
    from financial_dashboard.services.snapshots import upsert_account_snapshot

    snap_kind = {
        C.SNAPSHOT_ASSET: SnapshotKind.asset,
        C.SNAPSHOT_LIABILITY: SnapshotKind.liability,
    }
    for snap in scenario.account_snapshots:
        await upsert_account_snapshot(
            session,
            account_id=snap.account_pk,
            kind=snap_kind[snap.kind],
            category=(
                SnapshotCategory.bank_balance
                if snap.kind == C.SNAPSHOT_ASSET
                else SnapshotCategory.cc_outstanding
            ),
            as_of_date=snap.as_of,
            value=snap.current,
            source=SnapshotSource.manual,
            currency=getattr(snap, "currency", "INR"),
        )


async def _load_fidelity_transactions(
    session: AsyncSession,
    scenario: Scenario,
    fidelity_txn_count: int,
    stats: LoaderStats,
) -> None:
    """Route the first ``fidelity_txn_count`` transactions through the real
    merge/link path. Idempotent: an email whose ``message_id`` already exists
    is skipped, so its transaction is never re-created."""
    from sqlalchemy import select

    from financial_dashboard.services.linker import build_link_context, link_transaction
    from financial_dashboard.services.txn_merge import merge_transaction

    ctx = await build_link_context(session)

    # Pre-insert the emails/SMS for the fidelity subset so merge_transaction
    # has source rows to attach to. Emails use explicit PKs; rerun skips by
    # message_id.
    fidelity_txns = list(scenario.transactions[:fidelity_txn_count])
    email_pks: dict[str, int] = {}
    sms_pks: dict[str, int] = {}
    newly_inserted_emails: set[int] = set()
    created_txns = 0

    # First pass: ensure emails + sms rows exist for the fidelity subset.
    for t in fidelity_txns:
        existing = (
            await session.execute(
                select(Email.id).where(Email.message_id == _message_id(t.stable_id))
            )
        ).scalar_one_or_none()
        if existing is None:
            email = Email(
                id=t.email_pk,
                provider="gmail",
                message_id=_message_id(t.stable_id),
                source_id=1,
                sender=f"alerts@{t.bank}.bank.in",
                subject=f"{t.bank} {t.email_type}",
                received_at=_received_at(t, scenario.as_of),
                status="parsed",
            )
            session.add(email)
            email_pks[t.stable_id] = t.email_pk
            newly_inserted_emails.add(t.email_pk)
        else:
            email_pks[t.stable_id] = existing

        # SMS rows are loaded structurally (see _load_structural); we just
        # record the pk here so the second pass can attach it during the merge.
        if t.sms_pk is not None:
            sms_pks[t.stable_id] = t.sms_pk
    await session.flush()

    # Second pass: merge each fidelity transaction. On a fresh run this creates
    # the row; on a rerun the email pre-existed so we skip entirely (idempotent).
    enriched_via_sms = 0
    for t in fidelity_txns:
        if t.email_pk not in newly_inserted_emails:
            continue  # email pre-existed → transaction already present
        txn_data = _txn_data(t)
        # Pre-link using the linker so account/card resolution is exercised.
        proxy = _proxy_txn(t)
        link_transaction(ctx, proxy)
        if proxy.account_id is not None:
            txn_data["account_id"] = proxy.account_id
        if proxy.card_id is not None:
            txn_data["card_id"] = proxy.card_id
        elif t.card_pk is not None:
            # The linker resolves cards by mask; a row that carries an explicit
            # card_id but no mask (e.g. a bank-side CC bill payment that names
            # the card it pays) is honored directly so the projection's
            # exact-match card resolution can resolve it.
            txn_data["card_id"] = t.card_pk
        result = await merge_transaction(
            session, "email", txn_data, email_id=email_pks[t.stable_id]
        )
        if result.outcome == "created":
            created_txns += 1
        # merge_transaction creates the row from ``channel`` + ``txn_data`` only
        # — it deliberately does not carry category/category_method (those are a
        # categorizer concern). The scenario owns the intended category, so stamp
        # it here along with ``category_method='synthetic'`` so a fidelity row is
        # indistinguishable from its bulk-lane twin on every metadata axis.
        _stamp_category(result.transaction, t)
        # Paired (dedup) case: the same event arriving via SMS too. Re-running
        # it through the SMS channel should ENRICH the email-created row, never
        # create a second one — that is the cross-channel merge invariant.
        if t.sms_pk is not None and t.stable_id in sms_pks:
            sms_result = await merge_transaction(
                session,
                "sms",
                _txn_data(t),
                sms_message_id=sms_pks[t.stable_id],
            )
            if sms_result.outcome == "enriched":
                enriched_via_sms += 1
            # Keep the SmsMessage.transaction_id reverse link consistent with
            # the forward sms_message_id merge_transaction just set.
            if sms_result.transaction is not None:
                await session.execute(
                    SmsMessage.__table__.update()
                    .where(SmsMessage.__table__.c.id == sms_pks[t.stable_id])
                    .values(transaction_id=sms_result.transaction.id)
                )
    stats["fidelity_transactions"] = created_txns
    stats["fidelity_sms_enriched"] = enriched_via_sms


def _stamp_category(row, t) -> None:
    """Stamp the scenario's intended ``category`` + ``category_method`` onto a
    fidelity-lane row created by ``merge_transaction``.

    ``merge_transaction`` creates the row from ``channel`` + ``txn_data`` only —
    it deliberately does not carry category (a categorizer concern). The scenario
    owns the intended category, so the loader stamps it here.

    The one exception: the production self-transfer reference rule
    (``apply_reference_self_transfer_rule``, fired inside ``merge_transaction``)
    may already have categorized a paired self-transfer leg to
    ``category='self_transfer', category_method='rule'``. That is the real
    categorization rule doing its job in the fidelity lane, so it is preserved —
    the stamp only fills in rows the merge path left uncategorized.

    ``review_status`` and statement links are always carried through (they are
    not a categorizer concern). The category provenance fields
    (``category_method``/``category_confidence``/``category_model``) and
    ``review_reason`` are carried through when the scenario sets them so the
    fidelity lane reproduces the bulk lane's metadata axis exactly. Idempotent
    on the same values."""
    if row is None:
        return
    if row.category is None and t.category is not None:
        row.category = t.category
        row.category_method = t.category_method or "synthetic"
    elif t.category_method is not None and row.category is not None:
        # The merge path may have categorized via a production rule; the
        # scenario's method is still informative for the rows it owns.
        if row.category_method is None:
            row.category_method = t.category_method
    if t.category_method is not None and row.category_method is None:
        row.category_method = t.category_method
    if t.category_confidence is not None:
        row.category_confidence = t.category_confidence
    if t.category_model is not None:
        row.category_model = t.category_model
    if t.review_status is not None:
        row.review_status = t.review_status
    if t.review_reason is not None:
        row.review_reason = t.review_reason
    if t.statement_upload_id is not None:
        row.statement_upload_id = t.statement_upload_id
    if t.bank_statement_upload_id is not None:
        row.bank_statement_upload_id = t.bank_statement_upload_id


# ---------------------------------------------------------------------------
# Lane 2: bulk transactions (chunked Core inserts)
# ---------------------------------------------------------------------------


async def _load_bulk_transactions(
    session: AsyncSession,
    scenario: Scenario,
    fidelity_txn_count: int,
    stats: LoaderStats,
) -> None:
    """Insert the high-volume remainder via chunked ``executemany``."""
    bulk_txns = list(scenario.transactions[fidelity_txn_count:])
    if not bulk_txns:
        stats["bulk_transactions"] = 0
        stats["bulk_emails"] = 0
        return

    # Emails for bulk txns (chunked).
    email_rows = [
        {
            "id": t.email_pk,
            "provider": "gmail",
            "message_id": _message_id(t.stable_id),
            "source_id": 1,
            "sender": f"alerts@{t.bank}.bank.in",
            "subject": f"{t.bank} {t.email_type}",
            "received_at": _received_at(t, scenario.as_of),
            "status": "parsed",
        }
        for t in bulk_txns
    ]
    inserted_emails = await _chunked_insert(session, Email.__table__, email_rows)
    stats["bulk_emails"] = inserted_emails

    txn_rows = [
        {
            "id": bulk_txn_pk(i),
            "email_id": t.email_pk,
            # Bulk rows carry their SMS link when the scenario paired one, so a
            # bulk-lane paired event is indistinguishable from a fidelity one and
            # the SmsMessage.transaction_id reverse link can be set consistently.
            "sms_message_id": t.sms_pk,
            "account_id": t.account_pk,
            "card_id": t.card_pk,
            "bank": t.bank,
            "email_type": t.email_type,
            "direction": t.direction,
            "amount": t.amount,
            "currency": t.currency,
            "transaction_date": t.transaction_date,
            "transaction_time": t.transaction_time,
            "counterparty": t.counterparty,
            "card_mask": t.card_mask,
            "account_mask": t.account_mask,
            "reference_number": t.reference_number,
            "channel": t.channel,
            "balance": t.balance,
            "raw_description": t.raw_description,
            "category": t.category,
            # The scenario owns the category_method so the metadata axis
            # (manual/rule/llm/pending_llm/synthetic) is varied across the
            # corpus; ``None`` defaults to ``synthetic`` to preserve the
            # pre-expansion bulk-lane behaviour.
            "category_method": t.category_method or "synthetic",
            "category_confidence": t.category_confidence,
            "category_model": t.category_model,
            # Source is the scenario's truthful value: ``email`` for an
            # email-only row, ``sms+email`` for a paired (dedup) row.
            "source": t.source,
            "notified_channel": t.source,
            "review_status": t.review_status,
            "review_reason": t.review_reason,
            "statement_upload_id": t.statement_upload_id,
            "bank_statement_upload_id": t.bank_statement_upload_id,
        }
        for i, t in enumerate(bulk_txns)
    ]
    inserted_txns = await _chunked_insert(session, Transaction.__table__, txn_rows)
    stats["bulk_transactions"] = inserted_txns

    # Keep the SmsMessage.transaction_id reverse link consistent with the
    # Transaction.sms_message_id forward link we just wrote: every bulk-lane
    # row that carries an SMS now points back from that SMS to the transaction.
    # Idempotent — a rerun inserts zero txns and the UPDATE is a no-op then.
    sms_links = {
        t.sms_pk: bulk_txn_pk(i)
        for i, t in enumerate(bulk_txns)
        if t.sms_pk is not None
    }
    if sms_links:
        for sms_id, txn_id in sms_links.items():
            await session.execute(
                SmsMessage.__table__.update()
                .where(SmsMessage.__table__.c.id == sms_id)
                .values(transaction_id=txn_id)
            )


# ---------------------------------------------------------------------------
# Shared low-level helpers
# ---------------------------------------------------------------------------


async def _upsert(
    session: AsyncSession,
    model,
    *,
    _pk_name: str = "id",
    **fields,
) -> None:
    """Insert a structural row, ignoring a pre-existing primary key."""
    table = model.__table__
    stmt = sqlite_insert(table).values(**fields).on_conflict_do_nothing()
    await session.execute(stmt)


async def _chunked_insert(session: AsyncSession, table, rows: list[dict]) -> int:
    """Insert ``rows`` in chunks with ON CONFLICT DO NOTHING.

    The chunk size is capped by SQLite's per-statement variable limit (999 by
    default): each row contributes one bind per column, so the chunk size is
    ``max(1, 900 // num_columns)`` and never above :data:`CHUNK_SIZE`.

    Returns the number of rows that actually landed (rows in - rows skipped).
    For a same-seed rerun every chunk conflicts and zero rows land.
    """
    if not rows:
        return 0
    num_cols = len(rows[0])
    # Leave headroom under SQLite's default 999-variable ceiling.
    var_budget = 900
    adaptive = max(1, var_budget // max(num_cols, 1))
    chunk_size = min(CHUNK_SIZE, adaptive)
    total_landed = 0
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        before = (
            await session.execute(text(f"SELECT COUNT(*) FROM {table.name}"))
        ).scalar_one()
        stmt = sqlite_insert(table).values(chunk).on_conflict_do_nothing()
        await session.execute(stmt)
        after = (
            await session.execute(text(f"SELECT COUNT(*) FROM {table.name}"))
        ).scalar_one()
        total_landed += max(0, after - before)
    return total_landed


def _message_id(stable_txn_id: str) -> str:
    return f"<{stable_txn_id}@synthetic.local>"


def _received_at(t, as_of: datetime.date) -> datetime.datetime:
    """The email received_at for a transaction. Falls back to noon on ``as_of``
    when the row is undated (``transaction_date is None``) — an undated
    transaction is a legitimate row shape the cashflow report's Undated footnote
    counts, and its source email still needs a timestamp."""
    if t.transaction_date is not None:
        return datetime.datetime.combine(
            t.transaction_date, t.transaction_time or datetime.time(9, 0)
        )
    return datetime.datetime.combine(as_of, datetime.time(12, 0))


def _txn_data(t) -> dict:
    return {
        "bank": t.bank,
        "email_type": t.email_type,
        "direction": t.direction,
        "amount": t.amount,
        "currency": t.currency,
        "transaction_date": t.transaction_date,
        "transaction_time": t.transaction_time,
        "counterparty": t.counterparty,
        "card_mask": t.card_mask,
        "account_mask": t.account_mask,
        "reference_number": t.reference_number,
        "channel": t.channel,
        "balance": t.balance,
        "raw_description": t.raw_description,
    }


def _proxy_txn(t):
    """A lightweight stand-in carrying the linker-relevant fields, so the
    linker mutates something cheap before we copy the resolved ids back."""
    from financial_dashboard.db.models import Transaction

    return Transaction(
        bank=t.bank,
        email_type=t.email_type,
        direction=t.direction,
        amount=t.amount,
        card_mask=t.card_mask,
        account_mask=t.account_mask,
    )


async def count_rows(db_path: str | Path) -> dict[str, int]:
    """Recompute per-table row counts from the synthetic DB.

    A *missing* table (the DB predates a new table) reads as 0 — that is a
    legitimate backward-compat case for an older synthetic DB. But a *real*
    error (a malformed query, a locked DB, a corrupt page) is **never** silently
    coerced to zero: swallowing it would let manifest drift pass verification
    undetected. Only SQLite's ``no such table`` (error code 1 / message) is
    treated as "absent → 0"; everything else re-raises so a broken count is
    loud."""
    path = assert_synthetic_db_path(db_path)
    engine = create_async_engine(_db_url(path), echo=False)
    raw_tables = (
        "accounts",
        "cards",
        "categories",
        "email_sources",
        "fetch_rules",
        "emails",
        "sms_messages",
        "transactions",
        "manual_items",
        "cas_uploads",
        "statement_uploads",
        "bank_statement_uploads",
        "balance_snapshots",
        "snapshot_holdings",
        "investment_lots",
        "extension_runs",
    )
    raw: dict[str, int] = {}
    out: dict[str, int] = {}
    try:
        async with engine.connect() as conn:
            for tbl in raw_tables:
                try:
                    raw[tbl] = (
                        await conn.execute(text(f"SELECT COUNT(*) FROM {tbl}"))
                    ).scalar_one()
                except Exception as exc:  # noqa: BLE001 — narrow below
                    msg = str(exc).lower()
                    # Only a genuinely absent table is "0"; any other failure
                    # (locked DB, disk I/O, malformed schema) must propagate.
                    if "no such table" in msg:
                        raw[tbl] = 0
                    else:
                        raise
    finally:
        await engine.dispose()
    # Keys aligned with Scenario.counts() so verify can compare directly.
    aligned_keys = (
        "accounts",
        "cards",
        "categories",
        "email_sources",
        "fetch_rules",
        "emails",
        "sms_messages",
        "transactions",
        "manual_items",
        "cas_uploads",
        "investment_lots",
        "extension_runs",
    )
    for key in aligned_keys:
        out[key] = raw.get(key, 0)
    # statement_uploads is the combined CC + bank statement count (the scenario
    # does not distinguish the two tables).
    out["statement_uploads"] = raw.get("statement_uploads", 0) + raw.get(
        "bank_statement_uploads", 0
    )
    # Carry the raw counts through for diagnostics.
    out["_raw"] = raw
    return out


# Public re-exports for tests / CLI.
__all__ = [
    "CHUNK_SIZE",
    "LoaderStats",
    "count_rows",
    "create_synthetic_engine",
    "drop_synthetic_db",
    "load_scenario",
]

# Silence the Decimal import linter (used by type-checkers reading signatures).
_ = Decimal
