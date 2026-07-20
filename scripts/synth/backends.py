"""Backend-specific projection corpora, rendered by the **production** renderer.

Unlike :mod:`scripts.synth.paisa` (a pure, hand-reviewable Paisa-container
corpus that never imports the runtime), this module deliberately imports the
production Paisa renderer + investment classifier so the ledger / hledger /
beancount corpora it emits are produced by the *same code* the dashboard runs at
sync time — not a drifting duplicate. The document assembly here is a thin,
deterministic mapping from a :class:`Scenario` to a :class:`LedgerDocument`; the
per-backend text is 100% the production renderer's output.

This mirrors the loader's stance (the one other ``scripts.synth`` module that
imports the runtime): it is tooling, not a second implementation of projection.
It reads only — it never opens a session or writes a row.

Coverage (requirement: long names / multi-currency / lots are represented):
* a long, spaced account name that overruns the alignment column;
* a priced USD entry + its ``P``/``price`` directive (multi-currency);
* a complete investment lot, classified from the scenario's CAS payload by the
  real :func:`extract_lots_from_payload` (not re-implemented here).

Determinism: the scenario is deterministic, the slice is sorted and capped, and
the production renderer is pure — so the same ``(seed, as_of, profile)`` always
yields byte-identical backend journals.
"""

from decimal import Decimal
from typing import NamedTuple

from financial_dashboard.services.paisa.renderers import (
    normalize_default_account,
    render_document,
    validate_account_name,
)
from financial_dashboard.services.paisa.renderers.base import (
    EQUITY_OPENING,
    INR,
    InvestmentLotEntry,
    LedgerDocument,
    LedgerPosting,
    OpeningBalance,
    PriceDirective,
    ProjectedEntry,
    sanitize_commodity,
)
from scripts.synth import constants as C
from scripts.synth.models import Scenario, SynthTransaction

#: Cap so even a stress scenario yields a hand-reviewable corpus. The high-volume
#: rows stay in the SQLite DB (the loader's lane 2), not the rendered journal.
MAX_BACKEND_ENTRIES = 60

#: The artefact names this module produces, in a stable order. ``all`` renders
#: every supported backend; the production renderer registry is the source of
#: truth for the backend id set.
BACKEND_IDS: tuple[str, ...] = ("ledger", "hledger", "beancount")


class BackendCorpus(NamedTuple):
    """The rendered backend journals as in-memory bytes, ready to write."""

    artefacts: dict[str, bytes]
    entries: int
    backends: tuple[str, ...]


def _ledger_leaf(account) -> str:
    return account.label.split("(")[0].strip().replace(" ", "")


def _usd_rate_on(scenario: Scenario, on) -> Decimal | None:
    """The latest configured USD rate effective on/before ``on`` (or None).

    Mirrors ``PaisaProjectionConfig.fx_rate_for`` over the scenario's small,
    sorted-by-construction rate list, so the corpus's price directive agrees
    with what the production projection would emit.
    """
    chosen = None
    for fx in scenario.fx_rates:
        if fx.currency.upper() != "USD":
            continue
        if fx.date <= on and (chosen is None or fx.date > chosen.date):
            chosen = fx
    return chosen.rate if chosen is not None else None


def _lot_entries_from_scenario(
    scenario: Scenario,
) -> tuple[tuple[InvestmentLotEntry, ...], tuple[PriceDirective, ...]]:
    """Classify the scenario's CAS MF facts into lot entries via the *real*
    investment service, then build renderer lot entries + price directives.

    Reusing :func:`extract_lots_from_payload` guarantees the corpus's lots are
    exactly the ones CAS ingestion would persist — no second classification.
    """
    from financial_dashboard.services.investments import extract_lots_from_payload

    entries: list[InvestmentLotEntry] = []
    prices: list[PriceDirective] = []
    for cas in scenario.cas_uploads:
        lots, _exclusions = extract_lots_from_payload(cas.raw_payload)
        for lot in lots:
            instrument = sanitize_commodity(lot.instrument_id)
            entries.append(
                InvestmentLotEntry(
                    instrument=lot.instrument_id,
                    instrument_name=lot.instrument_name,
                    quantity=lot.quantity,
                    unit_cost=lot.unit_cost,
                    cost_basis=lot.cost_basis,
                    currency=lot.currency,
                    acquired_on=lot.acquired_on,
                )
            )
            # The lot's explicit per-unit CAS nav as an as-of price fact — a
            # truthful cost, not a synthesized market quote.
            prices.append(
                PriceDirective(
                    date=lot.acquired_on,
                    currency=instrument,
                    rate=lot.unit_cost,
                    unit=lot.currency,
                )
            )
    return tuple(entries), tuple(prices)


def _build_document(scenario: Scenario, max_entries: int) -> tuple[LedgerDocument, int]:
    """Assemble a representative :class:`LedgerDocument` from ``scenario``.

    The document is a deterministic, capped slice: every active bank account's
    opening balance, a date-ordered run of INR transactions, the priced USD
    transaction (multi-currency), the long/spaced insurance account, and the
    complete investment lot(s). The production renderer turns this into text.
    """
    active_banks = [
        a for a in scenario.accounts if a.type == C.BANK_ACCOUNT and a.active
    ]
    # Cutover = one day before the earliest transaction date, matching the
    # hand-reviewable corpus's convention so openings predate every entry.
    dates = [t.transaction_date for t in scenario.transactions if t.transaction_date]
    cutover = (
        (min(dates) - __import__("datetime").timedelta(days=1))
        if dates
        else scenario.as_of
    )

    openings: list[OpeningBalance] = [
        OpeningBalance(
            account_id=a.pk,
            account_name=f"Assets:Bank:{_ledger_leaf(a)}",
            amount=Decimal("5000.00"),
            source="snapshot",
            as_of=cutover,
        )
        for a in active_banks
    ]

    # Representative entries: the priced USD row (multi-currency), the long
    # insurance-premium row (long/spaced name), and a date-ordered slice of
    # ordinary INR rows. The USD row contributes a price directive.
    usd_txn = next(
        (t for t in scenario.transactions if t.currency == "USD" and t.ledger_account),
        None,
    )
    long_txn = next(
        (
            t
            for t in scenario.transactions
            if t.ledger_account and len(t.ledger_account) >= 47
        ),
        None,
    )

    entries: list[ProjectedEntry] = []
    prices: list[PriceDirective] = []
    chosen_ids: set[int] = set()

    def _add_entry(t: SynthTransaction, *, commodity: str, with_price: bool) -> None:
        if t is None:
            return
        sign = 1 if t.direction == C.DIRECTION_CREDIT else -1
        contra_sign = -sign
        # Income is credit-normal: invert so the entry balances to zero.
        if (t.ledger_account or "").startswith("Income"):
            sign, contra_sign = -contra_sign, -sign
        entries.append(
            ProjectedEntry(
                date=t.transaction_date,
                payee=t.counterparty or "Synthetic",
                txn_ids=(len(entries) + 1,),
                postings=(
                    LedgerPosting(
                        account=t.ledger_account,
                        amount=(t.amount * sign).quantize(Decimal("0.01")),
                        commodity=commodity,
                    ),
                    LedgerPosting(
                        account=t.ledger_counterpart,
                        amount=(t.amount * contra_sign).quantize(Decimal("0.01")),
                        commodity=commodity,
                    ),
                ),
                note=f"synthetic {t.email_type}",
                currency=commodity,
            )
        )
        chosen_ids.add(id(t))
        if with_price and commodity != INR:
            rate = _usd_rate_on(scenario, t.transaction_date)
            if rate is not None:
                prices.append(
                    PriceDirective(
                        date=t.transaction_date, currency=commodity, rate=rate
                    )
                )

    if usd_txn is not None:
        _add_entry(usd_txn, commodity="USD", with_price=True)
    if long_txn is not None:
        _add_entry(long_txn, commodity=INR, with_price=False)

    # Fill the remaining budget with a date-ordered slice of INR rows, skipping
    # the two special rows already added and any self-transfer pairs (which the
    # production projection collapses; a corpus is clearer without half-pairs).
    inr_pool = [
        t
        for t in scenario.transactions
        if t.currency == "INR"
        and t.transaction_date is not None
        and t.ledger_account
        and t.ledger_counterpart
        and (t.category or "") != "self_transfer"
        and id(t) not in chosen_ids
    ]
    inr_pool.sort(key=lambda t: (t.transaction_date, t.stable_id))
    for t in inr_pool[: max(0, max_entries - len(entries))]:
        _add_entry(t, commodity=INR, with_price=False)

    entries.sort(key=lambda e: (e.date, e.txn_ids[0] if e.txn_ids else 0))

    lot_entries, lot_prices = _lot_entries_from_scenario(scenario)
    prices.extend(lot_prices)

    # Deduplicate price directives by (date, currency), deterministic sort —
    # same rule the production projection applies before handing to the renderer.
    seen: dict[tuple, PriceDirective] = {}
    for price in prices:
        key = (price.date, price.currency)
        seen.setdefault(key, price)
    deduped_prices = tuple(sorted(seen.values(), key=lambda p: (p.date, p.currency)))

    # Declared accounts: every account named in a posting/opening. These are the
    # projection's *default* names (ledger-style, spaces legal); the per-backend
    # normalizer (beancount PascalCases them) is applied in _for_backend. Lot
    # accounts are derived per-backend by the renderer itself, so they are not
    # added here — the ledger family declares only what postings reference.
    declared: dict[str, None] = {}
    for ob in openings:
        declared.setdefault(ob.account_name, None)
    for e in entries:
        for p in e.postings:
            declared.setdefault(p.account, None)
    declared.setdefault(EQUITY_OPENING, None)

    doc = LedgerDocument(
        cutover_date=cutover,
        openings=tuple(openings),
        entries=tuple(entries),
        accounts_declared=tuple(sorted(declared)),
        price_directives=deduped_prices,
        lot_postings=lot_entries,
    )
    return doc, len(entries)


def _for_backend(doc: LedgerDocument, backend: str) -> LedgerDocument:
    """Return ``doc`` with every account name normalized for ``backend``.

    Every name here is a projection *default* (no operator overrides), so the
    production ``normalize_default_account`` is the correct transform — exactly
    what the projection applies via ``_finalize_name``. Validates each result
    against the backend's grammar so an incompatible name surfaces here rather
    than as a malformed file. Lot postings are left untouched: the renderer
    derives their (backend-legal, ISIN-based) accounts itself.
    """
    if backend == "ledger":
        # The ledger family's normalizer is identity; still validate names so a
        # malformed default is caught at generation time, not in the container.
        for name in doc.accounts_declared:
            validate_account_name(name, backend)
        return doc

    def _norm(name: str) -> str:
        return validate_account_name(normalize_default_account(name, backend), backend)

    openings = tuple(
        ob._replace(account_name=_norm(ob.account_name)) for ob in doc.openings
    )
    entries = tuple(
        e._replace(
            postings=tuple(p._replace(account=_norm(p.account)) for p in e.postings)
        )
        for e in doc.entries
    )
    declared = tuple(sorted({_norm(n) for n in doc.accounts_declared}))
    return doc._replace(openings=openings, entries=entries, accounts_declared=declared)


def build_backend_corpora(
    scenario: Scenario,
    *,
    backends: tuple[str, ...] = BACKEND_IDS,
    max_entries: int = MAX_BACKEND_ENTRIES,
) -> BackendCorpus:
    """Render deterministic ledger/hledger/beancount journals for ``scenario``.

    The text is the production renderer's exact output; only the document
    assembly is scenario-specific. Returns the journals keyed
    ``<backend>.journal`` plus a ``backends.meta.json`` describing the inputs,
    so a manifest can checksum them alongside the hand-reviewable corpus.
    """
    import json

    doc, entry_count = _build_document(scenario, max_entries)

    artefacts: dict[str, bytes] = {}
    for backend in backends:
        text = render_document(_for_backend(doc, backend), backend)
        artefacts[f"{backend}.journal"] = text.encode("utf-8")

    meta = {
        "backends": list(backends),
        "entry_count": entry_count,
        "lot_count": len(doc.lot_postings),
        "price_directive_count": len(doc.price_directives),
        "cutover_date": doc.cutover_date.isoformat() if doc.cutover_date else None,
        "currencies": sorted(
            {p.commodity for e in doc.entries for p in e.postings}
            | {pr.currency for pr in doc.price_directives}
        ),
        "as_of": scenario.as_of.isoformat(),
        "profile": scenario.profile,
        "seed": scenario.seed,
        # NOTE: deliberately no timestamp — the meta is part of the checksummed
        # corpus and must be byte-stable across runs (no drifting duplicate).
        "rendered_by": "financial_dashboard.services.paisa.renderers.render_document",
    }
    artefacts["backends.meta.json"] = (
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return BackendCorpus(
        artefacts=artefacts, entries=entry_count, backends=tuple(backends)
    )
