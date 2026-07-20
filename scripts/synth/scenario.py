"""The deterministic seeded scenario graph.

A single function — :func:`build_scenario` — turns ``(seed, as_of, profile)``
into a fully populated :class:`Scenario`. The same inputs always produce a
byte-identical graph: a dedicated :class:`random.Random` instance (never the
global), :class:`decimal.Decimal` money (never float), UUIDv5 stable ids, and
sorted/stable emit order.

Coverage (every kind the task names):

* salaries, expenses, refunds, reversals, fees, cash withdrawals
* self-transfers (paired debit/credit by shared reference)
* credit-card purchases and credit-card payments
* missing reference numbers, backdated rows, non-INR currency, unknowns
* statement-overlap / dedup cases (one event arriving as both an email and an
  SMS that the loader must merge)
* CAS-like holdings/snapshots, manual assets/liabilities (incl. deactivated)
* stale/deactivated accounts, cards and email sources

The graph is lane-agnostic: the loader decides which transactions flow through
the real-service fidelity lane and which through the chunked bulk lane.
"""

import datetime
import random
from calendar import monthrange
from decimal import Decimal

from scripts.synth import constants as C
from scripts.synth.coverage import compute_coverage
from scripts.synth.ids import money, quantize, stable_id, txn_reference
from scripts.synth.models import (
    Scenario,
    SynthAccount,
    SynthAccountSnapshot,
    SynthCard,
    SynthCasUpload,
    SynthCategory,
    SynthEmail,
    SynthEmailSource,
    SynthFetchRule,
    SynthFxRate,
    SynthManualItem,
    SynthOrphanEmail,
    SynthSms,
    SynthStatementUpload,
    SynthTransaction,
)


class Profile:
    """Size knobs for a scenario profile.

    Beyond raw volume, each profile carries two realism levers the monthly
    generator consumes:

    * ``expense_low/high_paise`` — the amount band for the generic UPI expense
      volume. It is **profile-scaled** so a stress profile's 200k+ rows do not
      crush the bank-scope income/expense ratio: stress emits far more rows but
      each at a far smaller amount, keeping the ratio truthful (>= 1.0) while
      still exercising scale. The dashboard never reads these as a special
      population — they are ordinary small UPI debits.
    * ``opening_balance_paise`` — the starting available balance per active
      savings account, sized so the tracked running balance never goes negative
      on an active bank account (a documented realism invariant).

    Both are explicit, sanctioned profile-specific amount bands: the only way
    to keep 200k+ rows, every edge/category AND a positive income/expense ratio
    in one corpus.
    """

    def __init__(
        self,
        name: str,
        months: int,
        monthly_volume: int,
        fidelity_txns: int,
        *,
        expense_low_paise: int,
        expense_high_paise: int,
        opening_balance_paise: int,
    ) -> None:
        self.name = name
        self.months = months
        self.monthly_volume = monthly_volume
        self.fidelity_txns = fidelity_txns
        self.expense_low_paise = expense_low_paise
        self.expense_high_paise = expense_high_paise
        self.opening_balance_paise = opening_balance_paise

    @property
    def expected_transaction_floor(self) -> int:
        """A conservative lower bound used by profile-size tests."""
        return self.months * self.monthly_volume


PROFILES: dict[str, Profile] = {
    # Hand-reviewable fixture corpus: a single month, a handful of rows.
    "golden": Profile(
        "golden",
        months=1,
        monthly_volume=6,
        fidelity_txns=20,
        expense_low_paise=50_00,
        expense_high_paise=4_000_00,
        opening_balance_paise=5_00_000_00,
    ),
    # Hundreds of transactions.
    "smoke": Profile(
        "smoke",
        months=4,
        monthly_volume=90,
        fidelity_txns=220,
        expense_low_paise=20_00,
        expense_high_paise=1_500_00,
        opening_balance_paise=20_00_000_00,
    ),
    # Several thousand. Smaller per-row band than smoke: more rows, smaller
    # amounts, so salary + recurring inflows keep income/expense >= 1.1.
    "ci": Profile(
        "ci",
        months=12,
        monthly_volume=300,
        fidelity_txns=400,
        expense_low_paise=5_00,
        expense_high_paise=1_00_000,
        opening_balance_paise=60_00_000_00,
    ),
    # >= 200,000 transactions. Very small per-row expense band (₹0.50–₹40) so
    # the bank-scope income/expense ratio stays >= 1.0 despite the volume; large
    # opening balances so the tracked running balance never goes negative. This
    # is the explicit profile-specific amount-band tradeoff for 200k+ rows: the
    # stress profile is a scale/robustness test, not a realism showcase, so its
    # per-row amounts are far smaller than smoke/ci. Realistic amounts live in
    # the golden/smoke/ci profiles.
    "stress": Profile(
        "stress",
        months=24,
        monthly_volume=8500,
        fidelity_txns=300,
        expense_low_paise=500,
        expense_high_paise=4_000,
        opening_balance_paise=50_00_000_00,
    ),
}

DEFAULT_AS_OF = datetime.date(2026, 7, 15)


# Banks and their transaction email_types. The masks/numbers are synthetic but
# shaped like the real formats the dashboard's linker matches positionally.
_BANKS: tuple[tuple[str, str, str, str], ...] = (
    # (bank, account_type, account_number, label)
    ("hdfc", C.BANK_ACCOUNT, "501000123456", "HDFC Savings"),
    ("hdfc", C.BANK_ACCOUNT, "501000654321", "HDFC Salary"),
    ("icici", C.BANK_ACCOUNT, "012345678901", "ICICI Savings"),
    ("kotak", C.BANK_ACCOUNT, "1234567890", "Kotak Salary"),
    ("idfc", C.BANK_ACCOUNT, "10002345678901", "IDFC Savings (stale)"),
    ("hdfc", C.CREDIT_CARD, "XXXX XXXX XXXX 4242", "HDFC Millennia CC"),
    ("icici", C.CREDIT_CARD, "XXXX XXXX XXXX 0066", "ICICI Sapphiro CC"),
    ("slice", C.CREDIT_CARD, "XXXX XXXX XXXX 7788", "Slice CC"),
    ("axis", C.CREDIT_CARD, "XXXX XXXX XXXX 9090", "Axis CC (stale)"),
)

_DEBIT_CARD_MASK = "XX7890"

_MERCHANTS = (
    ("BIGBASKET", "groceries"),
    ("SWIGGY", "dining"),
    ("ZOMATO", "dining"),
    ("AMAZON", "shopping"),
    ("FLIPKART", "shopping"),
    ("INDIAN OIL", "fuel"),
    ("UBER", "transport"),
    ("OLA", "transport"),
    ("NETFLIX", "subscriptions"),
    ("JIO", "utilities"),
    ("BOOKMYSHOW", "entertainment"),
    ("APOLLO PHARMACY", "healthcare"),
    ("MAKEMYTRIP", "travel"),
    ("BIG BAZAAR", "groceries"),
    ("DMART", "groceries"),
)


def build_scenario(
    seed: int = C.DEFAULT_SEED,
    as_of: datetime.date | None = None,
    profile: str = C.DEFAULT_PROFILE,
) -> Scenario:
    """Build the canonical deterministic scenario for the given inputs."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    prof = PROFILES[profile]
    if as_of is None:
        as_of = DEFAULT_AS_OF

    # A dedicated Random instance seeded from (seed, profile, as_of). The
    # profile and as_of are folded in so the same seed yields independent but
    # still reproducible graphs across profiles and cut-off dates. random.Random
    # accepts a str seed (hashed internally), so we join the parts first.
    rng = random.Random(f"{seed}:{prof.name}:{as_of.isoformat()}")

    # --- Structural entities (small, deterministic, always the same shape) ---
    accounts = _build_accounts(rng)
    cards = _build_cards(rng, accounts)
    categories = _build_categories()
    sources = _build_email_sources()
    rules = _build_fetch_rules(sources)
    manual = _build_manual_items(rng, as_of)
    fx_rates = _build_fx_rates(as_of)
    cas = _build_cas_uploads(rng, as_of)

    # --- Transaction volume (the bulk of the graph) ---
    # Built before statements/snapshots so those can derive their amounts from
    # the scenario's tracked running balances instead of hardcoded patches.
    txns, emails, sms, balances, cc_outstanding = _build_transactions(
        rng, prof, as_of, accounts, cards
    )
    statements = _build_statement_uploads(
        rng, accounts, as_of, balances, cc_outstanding, txns
    )
    snapshots = _build_account_snapshots(
        accounts, prof, balances, cc_outstanding, as_of, txns
    )
    orphan_emails = _build_orphan_emails(rng, as_of, sources)

    scenario = Scenario(
        seed=seed,
        as_of=as_of,
        profile=profile,
        accounts=tuple(accounts),
        cards=tuple(cards),
        categories=tuple(categories),
        email_sources=tuple(sources),
        fetch_rules=tuple(rules),
        emails=tuple(emails),
        sms=tuple(sms),
        transactions=tuple(txns),
        manual_items=tuple(manual),
        cas_uploads=tuple(cas),
        statement_uploads=tuple(statements),
        fx_rates=tuple(fx_rates),
        investment_lots=_expected_investment_lots(cas),
        account_snapshots=tuple(snapshots),
        orphan_emails=tuple(orphan_emails),
    )
    # Coverage is a pure function of the built graph; attach it last so the
    # manifest (and verify_manifest) can police the required branch set.
    return scenario._replace(coverage=compute_coverage(scenario))


# ---------------------------------------------------------------------------
# Structural builders
# ---------------------------------------------------------------------------


def _build_accounts(rng: random.Random) -> list[SynthAccount]:
    out: list[SynthAccount] = []
    for pk, (bank, acct_type, number, label) in enumerate(_BANKS, start=1):
        active = "stale" not in label
        out.append(
            SynthAccount(
                stable_id=stable_id("account", str(pk)),
                pk=pk,
                bank=bank,
                label=label,
                type=acct_type,
                account_number=number,
                statement_password="synthetic" if acct_type == C.CREDIT_CARD else None,
                statement_password_hint="synthetic-hint"
                if acct_type == C.CREDIT_CARD
                else None,
                active=active,
            )
        )
    return out


def _build_cards(rng: random.Random, accounts: list[SynthAccount]) -> list[SynthCard]:
    out: list[SynthCard] = []
    pk = 1
    # A debit card on the primary savings account.
    primary_savings = next(a for a in accounts if a.label == "HDFC Salary")
    out.append(
        SynthCard(
            stable_id=stable_id("card", str(pk)),
            pk=pk,
            account_pk=primary_savings.pk,
            card_mask=_DEBIT_CARD_MASK,
            label="HDFC Debit",
            is_primary=True,
            active=True,
        )
    )
    pk += 1
    # One credit card per credit-card account.
    for acct in accounts:
        if acct.type != C.CREDIT_CARD:
            continue
        last4 = _mask(acct, 4)
        active = acct.active
        out.append(
            SynthCard(
                stable_id=stable_id("card", str(pk)),
                pk=pk,
                account_pk=acct.pk,
                card_mask=f"XXXX XXXX XXXX {last4}",
                label=acct.label,
                is_primary=True,
                active=active,
            )
        )
        pk += 1
    # An add-on card on the primary credit-card account (a second mask on the
    # same liability), so the linker + projection exercise the add-on shape.
    primary_cc_acct = next(a for a in accounts if a.label == "HDFC Millennia CC")
    out.append(
        SynthCard(
            stable_id=stable_id("card", "addon"),
            pk=pk,
            account_pk=primary_cc_acct.pk,
            card_mask="XXXX XXXX XXXX 1133",
            label="HDFC Millennia Add-on",
            is_primary=False,
            active=True,
        )
    )
    return out


def _build_categories() -> list[SynthCategory]:
    return [SynthCategory(slug=slug, active=True) for slug in C.SEED_CATEGORY_SLUGS]


def _build_email_sources() -> list[SynthEmailSource]:
    return [
        SynthEmailSource(
            stable_id=stable_id("source", "gmail"),
            pk=1,
            provider="gmail",
            label="Synthetic Gmail",
            account_identifier="synthetic@example.com",
            active=True,
        ),
        SynthEmailSource(
            stable_id=stable_id("source", "fastmail"),
            pk=2,
            provider="fastmail",
            label="Synthetic Fastmail",
            account_identifier="synthetic@fastmail.com",
            active=True,
        ),
        SynthEmailSource(
            stable_id=stable_id("source", "offline"),
            pk=3,
            provider="offline",
            label="Stale offline source",
            account_identifier=None,
            active=False,
        ),
    ]


def _build_fetch_rules(
    sources: list[SynthEmailSource],
) -> list[SynthFetchRule]:
    gmail_pk = sources[0].pk
    raw = (
        ("icici", "credit_cards@icicibank.com", None, C.EMAIL_KIND_TXN),
        ("hdfc", "alerts@hdfcbank.bank.in", None, C.EMAIL_KIND_TXN),
        (
            "hdfc",
            "Emailstatements.cards@hdfcbank.net",
            "statement",
            C.EMAIL_KIND_CC_STATEMENT,
        ),
        ("kotak", "BankAlerts@kotak.com", None, C.EMAIL_KIND_TXN),
        (
            "slice",
            "noreply@slice.bank.in",
            "credit card statement",
            C.EMAIL_KIND_CC_STATEMENT,
        ),
        ("axis", "cc.statements@axis.bank.in", "statement", C.EMAIL_KIND_CC_STATEMENT),
    )
    out: list[SynthFetchRule] = []
    for pk, (bank, sender, subject, kind) in enumerate(raw, start=1):
        out.append(
            SynthFetchRule(
                stable_id=stable_id("rule", str(pk)),
                pk=pk,
                provider="gmail",
                source_pk=gmail_pk,
                sender=sender,
                subject=subject,
                bank=bank,
                email_kind=kind,
                enabled=True,
            )
        )
    return out


def _build_manual_items(
    rng: random.Random, as_of: datetime.date
) -> list[SynthManualItem]:
    raw = (
        # (name, kind, category, value_paise, active)
        ("Apartment", C.MANUAL_ASSET, C.MANUAL_CAT_PROPERTY, 85_00_000_00, True),
        ("Gold coins", C.MANUAL_ASSET, C.MANUAL_CAT_GOLD, 3_25_000_00, True),
        ("EPF", C.MANUAL_ASSET, C.MANUAL_CAT_EPF_PPF, 12_40_000_00, True),
        ("Emergency cash", C.MANUAL_ASSET, C.MANUAL_CAT_CASH, 1_50_000_00, True),
        ("Home loan", C.MANUAL_LIABILITY, C.MANUAL_CAT_LOAN, 42_00_000_00, True),
        ("Closed car loan", C.MANUAL_LIABILITY, C.MANUAL_CAT_LOAN, 0, False),
        ("Other asset", C.MANUAL_ASSET, C.MANUAL_CAT_OTHER, 50_000_00, True),
    )
    out: list[SynthManualItem] = []
    for pk, (name, kind, cat, paise, active) in enumerate(raw, start=1):
        as_of_date = as_of - datetime.timedelta(days=rng.randint(1, 40))
        out.append(
            SynthManualItem(
                stable_id=stable_id("manual", str(pk)),
                pk=pk,
                name=name,
                kind=kind,
                category=cat,
                active=active,
                notes="synthetic manual item" if active else "deactivated",
                as_of_date=as_of_date,
                value=quantize(Decimal(paise) / Decimal(100)),
            )
        )
    return out


def _build_fx_rates(as_of: datetime.date) -> list[SynthFxRate]:
    """Deterministic historical FX rates (INR per unit), anchored to ``as_of``.

    The set is deliberately shaped so the projection's ``priced`` policy has both
    covered and uncovered cases:

    * USD carries several rates spanning the scenario window, so a USD txn dated
      inside it is covered and a backdated USD txn (before the first rate) is
      ``missing_fx_rate``.
    * EUR carries one early rate (covered).
    * GBP has no rate at all, so any GBP txn is ``missing_fx_rate``.

    Rates are quantized to 4 dp to mirror ``config._parse_fx_rates`` and keep the
    rendered price directive byte-stable.
    """
    q4 = Decimal("0.0001")

    def rate(value: str) -> Decimal:
        return (Decimal(value)).quantize(q4)

    # Dates relative to as_of so the scenario stays anchored regardless of the
    # chosen as_of (the contract probe / tests pin --now to the scenario as_of).
    return [
        SynthFxRate(
            date=as_of - datetime.timedelta(days=192),
            currency="USD",
            rate=rate("83.1000"),
        ),
        SynthFxRate(
            date=as_of - datetime.timedelta(days=66),
            currency="USD",
            rate=rate("84.2000"),
        ),
        SynthFxRate(
            date=as_of - datetime.timedelta(days=5),
            currency="USD",
            rate=rate("85.5000"),
        ),
        SynthFxRate(
            date=as_of - datetime.timedelta(days=192),
            currency="EUR",
            rate=rate("90.0000"),
        ),
    ]


def _expected_investment_lots(cas_uploads: list[SynthCasUpload]) -> int:
    """Number of ``investment_lots`` rows the loader will persist for these CAS
    uploads.

    Uses the *same* pure extraction rule the loader's ``ingest_cas_payload`` →
    ``create_investment_lots`` path uses (``extract_lots_from_payload``), so the
    manifest's expected count tracks the real ingestion behaviour rather than a
    hand-maintained copy that could drift. A loader regression that drops or
    duplicates lots then surfaces as a manifest-verify count mismatch.
    """
    from financial_dashboard.services.investments import extract_lots_from_payload

    return sum(
        len(extract_lots_from_payload(cas.raw_payload)[0]) for cas in cas_uploads
    )


def _build_cas_uploads(
    rng: random.Random, as_of: datetime.date
) -> list[SynthCasUpload]:
    """Three CAS statements spanning multiple PANs and depositories, with a
    reconciled + an unreconciled portfolio, so the loader exercises the CAS
    ingestion path, its NSDL/CDSL guard, and the net-worth unreconciled flag.

    * NSDL (PAN A) — reconciled, carries the complete MF lot + the
      disposal/incomplete/demat exclusion facts.
    * CDSL (PAN B) — reconciled, standalone.
    * NSDL (PAN C) — unreconciled (``portfolio_ok=False``), with a disposal +
      incomplete MF fact (no new complete lot, so the deterministic lot count
      stays at one).
    """
    out: list[SynthCasUpload] = []

    def _payload(
        depository: str,
        pan: str,
        period_end: datetime.date,
        holdings,
        transactions=None,
        *,
        portfolio_ok: bool = True,
        portfolio_delta: str = "0.00",
    ) -> dict:
        accounts_block = [
            {
                "depository": depository.upper(),
                "dp_id": "12088700",
                "client_id": "00000001",
                # Distinguish the portfolio label by source depository so the
                # NSDL and CDSL uploads are legible as separate portfolios.
                "dp_name": f"Synthetic {depository.upper()} DP",
                "total_value": str(sum((h[1] for h in holdings), Decimal("0"))),
                "holdings": [
                    {
                        "name": h[0],
                        "isin": f"INE000A0101{i}",
                        "asset_class": "equity",
                        "quantity": "100",
                        "price": str(h[1]),
                        "value": str(h[1]),
                        "flags": [],
                        "notes": None,
                    }
                    for i, h in enumerate(holdings)
                ],
            }
        ]
        total = sum((h[1] for h in holdings), Decimal("0"))
        return {
            "file": "synthetic.pdf",
            "meta": {
                "source": depository,
                "investor_name": "Synthetic Investor",
                "pan": pan,
                "statement_period_start": "2026-04-01",
                "statement_period_end": period_end.isoformat(),
                "generated_on": period_end.isoformat(),
            },
            "accounts": accounts_block,
            "folios": [],
            "transactions": list(transactions or []),
            "summary": {
                "asset_class_totals": {"Equity": str(total)},
                "grand_total": str(total),
            },
            "reconciliation": {
                "portfolio_ok": portfolio_ok,
                "portfolio_delta": portfolio_delta,
                "holdings": [],
                "warnings": [],
            },
        }

    # MF + demat transaction facts attached to the NSDL upload. Only the first
    # is a *complete* acquisition fact (units+nav+amount+date+isin, mutually
    # consistent) — the one CAS fact shape the investment service turns into an
    # InvestmentLot. The rest are deliberately incomplete so the loader + the
    # projection's excluded-reason diagnostic exercise every exclusion path:
    #
    # * demat movement → ``not_mutual_fund`` (CAS carries no cost for demat)
    # * MF redemption  → ``disposal_transaction`` (a disposal, not acquisition)
    # * MF purchase missing nav → ``missing_lot_facts``
    #
    # The complete-lot instrument (INE000A01020) carries NO redemption: the
    # projection conservatively suppresses every instrument whose preserved CAS
    # facts contain a free-standing disposal, so tying a redemption to the same
    # ISIN would suppress the lot and yield ``investment_lot_count=0``. The
    # disposal lives on its own instrument (INE000A01045) so the complete lot
    # is emitted while the disposal scenario is still exercised.
    #
    # Acquisition dates are anchored relative to as_of (inside the statement
    # window) so they stay deterministic and scenario-anchored.
    mf_acquired = (as_of - datetime.timedelta(days=120)).isoformat()
    nsdl_mf_transactions = [
        {
            "scope": "mf",
            "source_ref": "synth-mf/1",
            "date": mf_acquired,
            "description": "Synthetic Liquid Fund",
            "isin": "INE000A01020",
            "transaction_type": "purchase",
            "units": "500",
            "nav": "100.00",
            "amount": "50000.00",
            "reference": "MFPUR001",
        },
        {
            "scope": "demat",
            "source_ref": "synth-demat/1",
            "date": mf_acquired,
            "isin": "INE000A01030",
            "transaction_type": "purchase",
            "quantity": "10",
        },
        # Disposal on a DIFFERENT instrument than the complete lot, so the lot
        # is not suppressed (disposal_history_unresolved → INE000A01045 only).
        {
            "scope": "mf",
            "source_ref": "synth-mf/2",
            "date": mf_acquired,
            "description": "Synthetic Disposal Fund",
            "isin": "INE000A01045",
            "transaction_type": "redemption",
            "units": "-50",
            "nav": "102.00",
            "amount": "-5100.00",
            "reference": "MFRED001",
        },
        {
            "scope": "mf",
            "source_ref": "synth-mf/3",
            "date": mf_acquired,
            "description": "Synthetic Short Fund",
            "isin": "INE000A01040",
            "transaction_type": "purchase",
            "units": "100",
            "amount": "1000.00",
            "reference": "MFPUR002",
        },
    ]

    nsdl_holdings = (
        ("Synthetic Equity A", Decimal("125000.00")),
        ("Synthetic Equity B", Decimal("75000.00")),
    )
    cdsl_holdings = (("Synthetic ETF C", Decimal("50000.00")),)
    period_a = as_of - datetime.timedelta(days=75)
    period_b = as_of - datetime.timedelta(days=15)

    payload_a = _payload(
        C.DEPOSITORY_NSDL, "ABCSYNTH01F", period_a, nsdl_holdings, nsdl_mf_transactions
    )
    out.append(
        SynthCasUpload(
            stable_id=stable_id("cas", "nsdl"),
            pk=1,
            portfolio_key="ABCSYNTH01F",
            depository_source=C.DEPOSITORY_NSDL,
            investor_name="Synthetic Investor",
            statement_date=period_a,
            grand_total=Decimal("200000.00"),
            portfolio_ok=True,
            portfolio_delta=Decimal("0.00"),
            holdings=nsdl_holdings,
            raw_payload=payload_a,
        )
    )
    payload_b = _payload(C.DEPOSITORY_CDSL, "ABCSYNTH02F", period_b, cdsl_holdings)
    out.append(
        SynthCasUpload(
            stable_id=stable_id("cas", "cdsl"),
            pk=2,
            portfolio_key="ABCSYNTH02F",
            depository_source=C.DEPOSITORY_CDSL,
            investor_name="Synthetic Investor",
            statement_date=period_b,
            grand_total=Decimal("50000.00"),
            portfolio_ok=True,
            portfolio_delta=Decimal("0.00"),
            holdings=cdsl_holdings,
            raw_payload=payload_b,
        )
    )
    # A third PAN, unreconciled (portfolio_ok=False, nonzero delta). Its MF
    # facts are a disposal + an incomplete purchase (no new complete lot), so
    # the deterministic investment-lot count stays at one while the net-worth
    # surface sees an unreconciled portfolio.
    period_c = as_of - datetime.timedelta(days=45)
    cdsl_holdings_c = (("Synthetic Equity D", Decimal("40000.00")),)
    unrecon_mf = [
        {
            "scope": "mf",
            "source_ref": "synth-mf-c/redeem",
            "date": mf_acquired,
            "description": "Synthetic Tax Fund",
            "isin": "INE000A01050",
            "transaction_type": "redemption",
            "units": "-20",
            "nav": "110.00",
            "amount": "-2200.00",
            "reference": "MFREDC001",
        },
        {
            "scope": "mf",
            "source_ref": "synth-mf-c/incomplete",
            "date": mf_acquired,
            "description": "Synthetic Tax Fund",
            "isin": "INE000A01050",
            "transaction_type": "purchase",
            "units": "30",
            "amount": "3000.00",
            "reference": "MFPURC001",
        },
    ]
    payload_c = _payload(
        C.DEPOSITORY_NSDL,
        "ABCSYNTH03F",
        period_c,
        cdsl_holdings_c,
        unrecon_mf,
        portfolio_ok=False,
        portfolio_delta="1500.00",
    )
    out.append(
        SynthCasUpload(
            stable_id=stable_id("cas", "nsdl-unrecon"),
            pk=3,
            portfolio_key="ABCSYNTH03F",
            depository_source=C.DEPOSITORY_NSDL,
            investor_name="Synthetic Investor",
            statement_date=period_c,
            grand_total=Decimal("40000.00"),
            portfolio_ok=False,
            portfolio_delta=Decimal("1500.00"),
            holdings=cdsl_holdings_c,
            raw_payload=payload_c,
        )
    )
    return out


def _build_statement_uploads(
    rng: random.Random,
    accounts: list[SynthAccount],
    as_of: datetime.date,
    balances: dict[int, Decimal],
    cc_outstanding: dict[int, Decimal],
    txns: list[SynthTransaction],
) -> list[SynthStatementUpload]:
    """Multiple CC + bank statements so the loader exercises
    ``emit_cc_snapshot`` / ``emit_bank_snapshot`` across the full status matrix.

    Coverage (every payment/lifecycle shape the task names):

    * CC: ``unpaid`` (PDF), ``paid`` (email_summary), ``parse_error`` (PDF).
    * bank: ``parsed`` (matched) + a second ``parsed`` on a different account.

    Amounts are **derived from the scenario's tracked balances** (not hardcoded
    patches): the CC ``total_amount_due`` is the card's accumulated outstanding
    (purchases net of payments), and the bank ``closing_balance`` is that
    account's final available balance. The CC ``due_date`` is rendered in the
    ``DD/MM/YYYY`` shape the dashboard's ``parse_cc_date`` expects (and the
    production statement pipeline stores), not ISO ``YYYY-MM-DD``.

    A handful of recent transactions on each statement account are linked via
    ``statement_upload_id`` / ``bank_statement_upload_id`` and counted in the
    upload's ``matched_count``/``imported_count`` so the reconciliation surface
    sees non-zero figures without bypassing the truthful link model. The first
    CC and first bank uploads additionally carry ``reconciliation_data``
    produced by the **real** ``reconcile_statement`` / ``reconcile_bank_statement``
    services over scenario-derived rows (see :mod:`scripts.synth.reconcile`),
    not hand-invented counts.
    """
    from scripts.synth.reconcile import reconcile_bank_offline, reconcile_cc_offline

    cc_acct = next(a for a in accounts if a.label == "HDFC Millennia CC")
    cc_acct_2 = next(a for a in accounts if a.label == "ICICI Sapphiro CC")
    cc_acct_3 = next(a for a in accounts if a.label == "Slice CC")
    bank_acct = next(a for a in accounts if a.label == "HDFC Savings")
    bank_acct_2 = next(a for a in accounts if a.label == "HDFC Salary")
    period_end_date = as_of - datetime.timedelta(days=20)
    period_end = period_end_date.isoformat()
    period_lo = period_end_date - datetime.timedelta(days=30)
    window = (period_lo, period_end_date)

    # CC outstanding clamped to a non-negative minimum (the primary card is unpaid).
    cc_due_raw = cc_outstanding.get(cc_acct.pk, Decimal("0.00"))
    cc_due = max(cc_due_raw, Decimal("5000.00"))
    cc_min = (cc_due * Decimal("0.05")).quantize(Decimal("0.01"))
    cc_due_str = f"{cc_due:,.2f}"
    cc_min_str = f"{cc_min:,.2f}"
    # DD/MM/YYYY — the format cc-parser emits and parse_cc_date consumes.
    cc_due_date = (as_of + datetime.timedelta(days=15)).strftime("%d/%m/%Y")

    # The second card is fully paid down (zero outstanding) — a paid statement.
    cc2_paid = max(cc_outstanding.get(cc_acct_2.pk, Decimal("0.00")), Decimal("0.00"))
    cc2_due_str = f"{cc2_paid:,.2f}"
    cc2_min_str = f"{(cc2_paid * Decimal('0.05')).quantize(Decimal('0.01')):,.2f}"

    bank_closing = balances.get(bank_acct.pk, Decimal("0.00"))
    bank_closing_str = f"{bank_closing:,.2f}"
    bank2_closing = balances.get(bank_acct_2.pk, Decimal("0.00"))
    bank2_closing_str = f"{bank2_closing:,.2f}"

    # Link recent transactions on each statement account so the reconciliation
    # counts (matched/imported) are non-zero. Only rows already on the account
    # and dated within the statement window are eligible — a truthful subset.
    def _eligible(acct_pk: int, *, stmt_id: int, bank_stmt: bool) -> int:
        n = 0
        for idx, t in enumerate(txns):
            if t.account_pk != acct_pk or t.transaction_date is None:
                continue
            if not (period_lo <= t.transaction_date <= period_end_date):
                continue
            if n >= 8:
                break
            # Mutate the NamedTuple in place by replacing the list slot. The
            # transactions list is the single owner of these rows at build time.
            # Preserve an existing review_status (e.g. a row the scenario
            # deliberately marked resolved/notified); only uncategorized rows
            # are stamped ``reviewed`` by the statement-import link.
            new_status = t.review_status if t.review_status is not None else "reviewed"
            if bank_stmt:
                txns[idx] = t._replace(
                    bank_statement_upload_id=stmt_id, review_status=new_status
                )
            else:
                txns[idx] = t._replace(
                    statement_upload_id=stmt_id, review_status=new_status
                )
            n += 1
        return n

    cc_matched = _eligible(cc_acct.pk, stmt_id=1, bank_stmt=False)
    cc2_matched = _eligible(cc_acct_2.pk, stmt_id=2, bank_stmt=False)
    bank_matched = _eligible(bank_acct.pk, stmt_id=3, bank_stmt=True)
    bank2_matched = _eligible(bank_acct_2.pk, stmt_id=4, bank_stmt=True)

    # Real reconciliation_data via the production reconcile services (offline).
    cc_recon = reconcile_cc_offline(
        txns, account_pk=cc_acct.pk, window=window, period_end=period_end_date
    )
    bank_recon = reconcile_bank_offline(
        txns,
        account_pk=bank_acct.pk,
        window=window,
        closing_balance=bank_closing_str,
    )

    return [
        # CC #1 — unpaid, PDF, real reconciliation_data.
        SynthStatementUpload(
            stable_id=stable_id("stmt", "cc"),
            pk=1,
            account_pk=cc_acct.pk,
            email_pk=None,
            bank=cc_acct.bank,
            filename="synthetic-cc-statement.pdf",
            file_path="data/synthetic/statements/synthetic-cc-statement.pdf",
            source_kind=C.STMT_SOURCE_PDF,
            status=C.STMT_STATUS_PARSED,
            card_number=cc_acct.account_number,
            statement_name=cc_acct.label,
            due_date=cc_due_date,
            total_amount_due=cc_due_str,
            minimum_amount_due=cc_min_str,
            payment_status=C.PAYMENT_UNPAID,
            closing_balance=None,
            statement_period_end=period_end,
            parsed_txn_count=cc_matched,
            matched_count=cc_matched,
            imported_count=cc_matched,
            reconciliation_data=cc_recon,
            password_hint="synthetic-hint",
        ),
        # CC #2 — paid, sourced from an email summary (no PDF).
        SynthStatementUpload(
            stable_id=stable_id("stmt", "cc-paid"),
            pk=2,
            account_pk=cc_acct_2.pk,
            email_pk=None,
            bank=cc_acct_2.bank,
            filename="synthetic-cc-statement-paid.eml",
            file_path="data/synthetic/statements/synthetic-cc-statement-paid.eml",
            source_kind=C.STMT_SOURCE_EMAIL_SUMMARY,
            status=C.STMT_STATUS_IMPORTED,
            card_number=cc_acct_2.account_number,
            statement_name=cc_acct_2.label,
            due_date=cc_due_date,
            total_amount_due=cc2_due_str,
            minimum_amount_due=cc2_min_str,
            payment_status=C.PAYMENT_PAID,
            closing_balance=None,
            statement_period_end=period_end,
            parsed_txn_count=cc2_matched,
            matched_count=cc2_matched,
            imported_count=cc2_matched,
        ),
        # CC #3 — parse_error (a password-protected PDF that failed to parse).
        SynthStatementUpload(
            stable_id=stable_id("stmt", "cc-error"),
            pk=5,
            account_pk=cc_acct_3.pk,
            email_pk=None,
            bank=cc_acct_3.bank,
            filename="synthetic-cc-statement-error.pdf",
            file_path="data/synthetic/statements/synthetic-cc-statement-error.pdf",
            source_kind=C.STMT_SOURCE_PDF,
            status=C.STMT_STATUS_PARSE_ERROR,
            card_number=cc_acct_3.account_number,
            statement_name=cc_acct_3.label,
            due_date=None,
            total_amount_due=None,
            minimum_amount_due=None,
            payment_status=None,
            closing_balance=None,
            statement_period_end=period_end,
            parsed_txn_count=0,
            matched_count=0,
            imported_count=0,
            password_hint="synthetic-hint",
        ),
        # bank #1 — parsed, matched, real reconciliation_data.
        SynthStatementUpload(
            stable_id=stable_id("stmt", "bank"),
            pk=3,
            account_pk=bank_acct.pk,
            email_pk=None,
            bank=bank_acct.bank,
            filename="synthetic-bank-statement.pdf",
            file_path="data/synthetic/statements/synthetic-bank-statement.pdf",
            source_kind=C.STMT_SOURCE_PDF,
            status=C.STMT_STATUS_PARSED,
            card_number=None,
            statement_name=bank_acct.label,
            due_date=None,
            total_amount_due=None,
            minimum_amount_due=None,
            payment_status=None,
            closing_balance=bank_closing_str,
            statement_period_end=period_end,
            parsed_txn_count=bank_matched,
            matched_count=bank_matched,
            imported_count=bank_matched,
            reconciliation_data=bank_recon,
        ),
        # bank #2 — parsed on a different account (HDFC Salary).
        SynthStatementUpload(
            stable_id=stable_id("stmt", "bank-2"),
            pk=4,
            account_pk=bank_acct_2.pk,
            email_pk=None,
            bank=bank_acct_2.bank,
            filename="synthetic-bank-statement-2.pdf",
            file_path="data/synthetic/statements/synthetic-bank-statement-2.pdf",
            source_kind=C.STMT_SOURCE_PDF,
            status=C.STMT_STATUS_PARSED,
            card_number=None,
            statement_name=bank_acct_2.label,
            due_date=None,
            total_amount_due=None,
            minimum_amount_due=None,
            payment_status=None,
            closing_balance=bank2_closing_str,
            statement_period_end=period_end,
            parsed_txn_count=bank2_matched,
            matched_count=bank2_matched,
            imported_count=bank2_matched,
        ),
    ]


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------


def _month_dates(as_of: datetime.date, months: int) -> list[datetime.date]:
    """The first day of each of the trailing ``months`` months, oldest first,
    ending with the month containing ``as_of``."""
    from dateutil.relativedelta import relativedelta

    return [
        (as_of - relativedelta(months=back)).replace(day=1)
        for back in range(months - 1, -1, -1)
    ]


def _balance_at_cutover(
    account_pk: int,
    is_cc: bool,
    bank_opening: Decimal,
    txns: list[SynthTransaction],
    cutover: datetime.date,
) -> Decimal:
    """Replay a single account's INR-trackable transactions dated on/before the
    cutover to derive its truthful balance at the cutover.

    This is the value the projection strikes as the opening balance, so that
    ``opening + Σ(post-cutover emitted postings)`` equals the tracked running
    balance at every point (which the realism invariant keeps >= 0). Only INR
    (or NULL-currency) rows are folded in: foreign-currency rows post to a
    separate commodity and the projection skips invalid-currency rows, so
    neither belongs in the INR opening.

    * Bank (asset): start from ``bank_opening``; credits add, debits subtract.
    * Credit-card (liability): start from 0; debits (purchases) add to what is
      owed, credits (payments) subtract.
    """
    value = Decimal("0.00") if is_cc else bank_opening
    for t in txns:
        if t.account_pk != account_pk or t.transaction_date is None:
            continue
        if t.transaction_date > cutover:
            continue
        if (t.currency or "INR") != "INR":
            continue
        if is_cc:
            value += t.amount if t.direction == C.DIRECTION_DEBIT else -t.amount
        else:
            value += t.amount if t.direction == C.DIRECTION_CREDIT else -t.amount
    return quantize(value)


def _build_account_snapshots(
    accounts: list[SynthAccount],
    prof: Profile,
    balances: dict[int, Decimal],
    cc_outstanding: dict[int, Decimal],
    as_of: datetime.date,
    txns: list[SynthTransaction],
) -> list[SynthAccountSnapshot]:
    """Realistic opening/current balance snapshots for every active account,
    derived from the scenario's tracked running balance rather than a
    hardcoded patch.

    * Bank accounts: ``opening`` is the profile opening balance the running
      balance started from; ``current`` is the account's final tracked balance.
    * Credit-card accounts: ``opening`` is zero (no prior outstanding carried);
      ``current`` is the accumulated outstanding (purchases net of payments).

    The loader turns each into a ``BalanceSnapshot`` row (asset for bank,
    liability for card), so the net-worth surface has truthful, balance-derived
    figures instead of one giant manual snapshot.

    In addition to the current snapshot per active account, a trailing monthly
    history is emitted for the primary salary account so the net-worth *trend*
    sees multiple months (forward-fill across month boundaries), and one
    non-INR snapshot is emitted so the net-worth exclusion path (currency
    != INR) is exercised. A card with zero accumulated outstanding supplies the
    zero-CC snapshot shape.
    """
    out: list[SynthAccountSnapshot] = []
    bank_opening = quantize(Decimal(prof.opening_balance_paise) / Decimal(100))
    for acct in accounts:
        if not acct.active:
            continue
        if acct.type == C.BANK_ACCOUNT:
            out.append(
                SynthAccountSnapshot(
                    account_pk=acct.pk,
                    kind=C.SNAPSHOT_ASSET,
                    opening=bank_opening,
                    current=balances.get(acct.pk, bank_opening),
                    as_of=as_of,
                )
            )
        elif acct.type == C.CREDIT_CARD:
            out.append(
                SynthAccountSnapshot(
                    account_pk=acct.pk,
                    kind=C.SNAPSHOT_LIABILITY,
                    opening=Decimal("0.00"),
                    current=cc_outstanding.get(acct.pk, Decimal("0.00")),
                    as_of=as_of,
                )
            )

    # Truthful opening-balance snapshots at the documented projection cutover
    # (``PROJECTION_CUTOVER``), one per active asset/liability account. The
    # value is replayed from the scenario's tracked balances (every INR
    # transaction on the account dated on/before the cutover), so a projection
    # run with ``cutover_date = PROJECTION_CUTOVER`` opens each account at its
    # real point-in-time balance and the post-cutover running balance never
    # goes negative. Dated one day before the cutover so the projection's
    # ``as_of_date <= cutover`` lookup finds it and the cutover day's own
    # transactions are folded into the opening (the projection emits only
    # strictly-post-cutover rows).
    cutover_open_date = C.PROJECTION_CUTOVER - datetime.timedelta(days=1)
    for acct in accounts:
        if not acct.active:
            continue
        is_cc = acct.type == C.CREDIT_CARD
        if acct.type != C.BANK_ACCOUNT and not is_cc:
            continue
        opening_value = _balance_at_cutover(
            acct.pk, is_cc, bank_opening, txns, C.PROJECTION_CUTOVER
        )
        out.append(
            SynthAccountSnapshot(
                account_pk=acct.pk,
                kind=C.SNAPSHOT_LIABILITY if is_cc else C.SNAPSHOT_ASSET,
                opening=Decimal("0.00") if is_cc else bank_opening,
                current=opening_value,
                as_of=cutover_open_date,
                note="projection_cutover_opening",
            )
        )

    # Trailing monthly snapshots for the salary account so the net-worth trend
    # spans multiple months. Values are a deterministic forward walk from the
    # opening balance toward the final tracked balance — not new source facts,
    # just a truthful interpolation of the same balance the current snapshot
    # already reports.
    from dateutil.relativedelta import relativedelta

    salary = next((a for a in accounts if a.label == "HDFC Salary"), None)
    if salary is not None:
        final = balances.get(salary.pk, bank_opening)
        for back in (3, 2, 1):
            month_start = (as_of - relativedelta(months=back)).replace(day=1)
            frac = Decimal(back) / Decimal(4)
            val = quantize(bank_opening + (final - bank_opening) * frac)
            out.append(
                SynthAccountSnapshot(
                    account_pk=salary.pk,
                    kind=C.SNAPSHOT_ASSET,
                    opening=bank_opening,
                    current=val,
                    as_of=month_start,
                )
            )

    # A non-INR snapshot (USD) so the net-worth surface excludes it from the
    # INR totals. Dated off the ``as_of`` cutover so its
    # ``(account, category, as_of_date)`` upsert key does not overwrite the
    # salary account's real INR current snapshot; it remains a distinct row the
    # net-worth reader skips by currency (and, being superseded, by recency).
    if salary is not None:
        out.append(
            SynthAccountSnapshot(
                account_pk=salary.pk,
                kind=C.SNAPSHOT_ASSET,
                opening=Decimal("0.00"),
                current=Decimal("100.00"),
                as_of=as_of - datetime.timedelta(days=47),
                note="non_inr_excluded",
                currency="USD",
            )
        )
    return out


def _build_orphan_emails(
    rng: random.Random,
    as_of: datetime.date,
    sources: list[SynthEmailSource],
) -> list[SynthOrphanEmail]:
    """A small set of standalone source rows not tied to any transaction — a
    pending, a failed and a skipped email — so the review/error surfaces have
    non-empty states at every profile scale.

    These are loaded structurally (by ``message_id`` natural key) and never
    double-created by the transaction lanes, because their pks (9001+) sit well
    above the transaction-email range (100+). They keep count invariants intact:
    ``Scenario.counts()['emails']`` already folds them in alongside the
    transaction-linked emails.
    """
    gmail_pk = sources[0].pk
    # Deterministic, profile-independent offsets so the same (seed, as_of) gives
    # the same orphan set at every scale.
    base = as_of - datetime.timedelta(days=2)
    raw = (
        (
            "pending",
            "alerts@hdfc.bank.in",
            "hdfc account_debit_upi pending",
            "pending",
            None,
        ),
        (
            "failed",
            "alerts@icicibank.com",
            "icici cc_debit_purchase failed parse",
            "failed",
            "synthetic: parser could not extract amount",
        ),
        (
            "skipped",
            "noreply@slice.bank.in",
            "slice non-transaction newsletter",
            "skipped",
            "synthetic: non-transaction email",
        ),
    )
    out: list[SynthOrphanEmail] = []
    for i, (kind, sender, subject, status, error) in enumerate(raw):
        pk = 9001 + i
        sid = stable_id("orphan-email", kind)
        out.append(
            SynthOrphanEmail(
                stable_id=sid,
                pk=pk,
                provider="gmail",
                message_id=f"<{sid}@synthetic.local>",
                source_pk=gmail_pk,
                sender=sender,
                subject=subject,
                received_at=datetime.datetime.combine(
                    base - datetime.timedelta(days=i), datetime.time(9, 0)
                ),
                status=status,
                bank=kind,
                error=error,
            )
        )
    return out


def _build_transactions(
    rng: random.Random,
    prof: Profile,
    as_of: datetime.date,
    accounts: list[SynthAccount],
    cards: list[SynthCard],
) -> tuple[
    list[SynthTransaction],
    list[SynthEmail],
    list[SynthSms],
    dict[int, Decimal],
    dict[int, Decimal],
]:
    """Build the full transaction set plus the emails/SMS that sourced them.

    Returns the transactions, emails, SMS, the final per-account running
    balance (so statements/snapshots derive from it), and the per-credit-card
    outstanding (purchases net of payments) for the same reason.
    """
    txns: list[SynthTransaction] = []
    emails: list[SynthEmail] = []
    sms: list[SynthSms] = []

    savings = [a for a in accounts if a.type == C.BANK_ACCOUNT and a.active]
    salary_acct = next(a for a in accounts if a.label == "HDFC Salary")
    cc_accounts = [a for a in accounts if a.type == C.CREDIT_CARD and a.active]
    primary_cc = cc_accounts[0]

    # Running balance per savings account so the generated rows carry a
    # realistic available balance (the dashboard's merge logic keys on this).
    # Profile-scaled opening so an active bank account never tracks negative.
    opening = quantize(Decimal(prof.opening_balance_paise) / Decimal(100))
    balances: dict[int, Decimal] = {a.pk: opening for a in savings}
    # Credit-card outstanding accumulates purchases (debit) and is paid down by
    # card-side payments (credit). Used to derive a realistic CC statement.
    cc_outstanding: dict[int, Decimal] = {a.pk: Decimal("0.00") for a in cc_accounts}

    # email/sms pk counters (structural emails above used low pks; volume
    # emails start above 100 to leave headroom).
    email_pk = 100
    sms_pk = 100

    def add_email(bank: str, received_at: datetime.datetime, subject: str) -> int:
        nonlocal email_pk
        sid = stable_id("email", str(email_pk))
        e = SynthEmail(
            stable_id=sid,
            pk=email_pk,
            provider="gmail",
            message_id=f"<{sid}@synthetic.local>",
            source_pk=1,
            sender=f"alerts@{bank}.bank.in",
            subject=subject,
            received_at=received_at,
            status="parsed",
            bank=bank,
        )
        emails.append(e)
        email_pk += 1
        return e.pk

    def add_sms(bank: str, received_at: datetime.datetime, body: str) -> int:
        nonlocal sms_pk
        sid = stable_id("sms", str(sms_pk))
        s = SynthSms(
            stable_id=sid,
            pk=sms_pk,
            bank=bank,
            sender=f"{bank.upper()}-SMS",
            body=body,
            received_at=received_at,
            status="parsed",
        )
        sms.append(s)
        sms_pk += 1
        return s.pk

    def apply_balance(acct_pk: int, direction: str, amount: Decimal) -> Decimal:
        delta = amount if direction == C.DIRECTION_CREDIT else -amount
        balances[acct_pk] = quantize(balances.get(acct_pk, Decimal("0")) + delta)
        return balances[acct_pk]

    def apply_cc(acct_pk: int, direction: str, amount: Decimal) -> None:
        # A purchase (debit) raises what the card holder owes; a payment
        # (credit) pays it down. Tracked so the CC statement total is truthful.
        delta = amount if direction == C.DIRECTION_DEBIT else -amount
        cc_outstanding[acct_pk] = quantize(
            cc_outstanding.get(acct_pk, Decimal("0")) + delta
        )

    def push(
        *,
        bank: str,
        email_type: str,
        direction: str,
        amount: Decimal,
        txn_date: datetime.date | None,
        counterparty: str | None,
        category: str | None,
        source: str,
        account_pk: int | None,
        card_pk: int | None,
        card_mask: str | None,
        account_mask: str | None,
        reference_number: str | None,
        channel: str | None,
        balance: Decimal | None,
        currency: str = "INR",
        raw_description: str | None = None,
        ledger_account: str | None,
        ledger_counterpart: str | None,
        dedup_group: str | None = None,
        txn_time: datetime.time | None = None,
        review_status: str | None = None,
        statement_upload_id: int | None = None,
        bank_statement_upload_id: int | None = None,
        category_method: str | None = None,
        category_confidence: float | None = None,
        category_model: str | None = None,
        review_reason: str | None = None,
    ) -> None:
        # An undated row keeps ``transaction_date is None`` but its source email
        # still needs a timestamp — fall back to noon on ``as_of``.
        email_dt = (
            datetime.datetime.combine(txn_date, txn_time or datetime.time(9, 0))
            if txn_date is not None
            else datetime.datetime.combine(as_of, datetime.time(12, 0))
        )
        sid = stable_id(
            "txn",
            str(len(txns)),
            bank,
            str(amount),
            (txn_date.isoformat() if txn_date is not None else "undated"),
            str(reference_number),
        )
        email_pk_for_txn = add_email(
            bank,
            email_dt,
            f"{bank} {email_type} {counterparty or ''}".strip(),
        )
        txns.append(
            SynthTransaction(
                stable_id=sid,
                bank=bank,
                email_type=email_type,
                direction=direction,
                amount=amount,
                currency=currency,
                transaction_date=txn_date,
                transaction_time=txn_time,
                counterparty=counterparty,
                card_mask=card_mask,
                account_mask=account_mask,
                reference_number=reference_number,
                channel=channel,
                balance=balance,
                raw_description=raw_description,
                category=category,
                source=source,
                account_pk=account_pk,
                card_pk=card_pk,
                email_pk=email_pk_for_txn,
                sms_pk=None,
                dedup_group=dedup_group,
                ledger_account=ledger_account,
                ledger_counterpart=ledger_counterpart,
                review_status=review_status,
                statement_upload_id=statement_upload_id,
                bank_statement_upload_id=bank_statement_upload_id,
                category_method=category_method,
                category_confidence=category_confidence,
                category_model=category_model,
                review_reason=review_reason,
            )
        )

    for month_start in _month_dates(as_of, prof.months):
        # --- Salary (credit, monthly) ---
        amt = money(rng, 1_20_000_00, 1_80_000_00)
        bal = apply_balance(salary_acct.pk, C.DIRECTION_CREDIT, amt)
        d = _clamp(_pick_day(rng, month_start, 1), as_of)
        push(
            bank=salary_acct.bank,
            email_type="account_credit_salary",
            direction=C.DIRECTION_CREDIT,
            amount=amt,
            txn_date=d,
            counterparty="EMPLOYER INC",
            category="salary",
            source="email",
            account_pk=salary_acct.pk,
            card_pk=None,
            card_mask=None,
            account_mask=_mask(salary_acct, 6),
            reference_number=txn_reference(stable_id("txn-salary", d.isoformat())),
            channel="neft",
            balance=bal,
            ledger_account=f"Assets:Bank:{_ledger_leaf(salary_acct)}",
            ledger_counterpart="Income:Salary",
        )

        # --- Interest (credit, monthly) — recurring savings interest. -------
        interest = money(rng, 1_500_00, 7_500_00)
        interest_acct = rng.choice(savings)
        bal = apply_balance(interest_acct.pk, C.DIRECTION_CREDIT, interest)
        d = _clamp(_pick_day(rng, month_start, 2), as_of)
        push(
            bank=interest_acct.bank,
            email_type="account_credit_interest",
            direction=C.DIRECTION_CREDIT,
            amount=interest,
            txn_date=d,
            counterparty="BANK INTEREST",
            category="interest",
            source="email",
            account_pk=interest_acct.pk,
            card_pk=None,
            card_mask=None,
            account_mask=_mask(interest_acct, 6),
            reference_number=txn_reference(stable_id("txn-interest", d.isoformat())),
            channel="interest",
            balance=bal,
            ledger_account=f"Assets:Bank:{_ledger_leaf(interest_acct)}",
            ledger_counterpart="Income:Interest",
        )

        # --- Other / freelance income (credit, most months) ----------------
        # Skipped ~1 month in 5 so the inflow stream is realistic, not rigid.
        if rng.random() < 0.8:
            oinc = money(rng, 25_000_00, 1_10_000_00)
            oinc_acct = rng.choice(savings)
            bal = apply_balance(oinc_acct.pk, C.DIRECTION_CREDIT, oinc)
            d = _clamp(_pick_day(rng, month_start, rng.randint(8, 22)), as_of)
            push(
                bank=oinc_acct.bank,
                email_type="account_credit_income",
                direction=C.DIRECTION_CREDIT,
                amount=oinc,
                txn_date=d,
                counterparty="FREELANCE CLIENT",
                category="other_income",
                source="email",
                account_pk=oinc_acct.pk,
                card_pk=None,
                card_mask=None,
                account_mask=_mask(oinc_acct, 6),
                reference_number=txn_reference(
                    stable_id("txn-oincome", d.isoformat(), str(len(txns)))
                ),
                channel="neft",
                balance=bal,
                ledger_account=f"Assets:Bank:{_ledger_leaf(oinc_acct)}",
                ledger_counterpart="Income:Other Income",
            )

        # --- Repayment / transfers-in (credit, monthly) --------------------
        # Money handed back TO the account holder (somebody repaying them). This
        # is the transfers-in bucket, so it is a CREDIT — never a debit.
        rep = money(rng, 5_000_00, 35_000_00)
        rep_acct = rng.choice(savings)
        bal = apply_balance(rep_acct.pk, C.DIRECTION_CREDIT, rep)
        d = _clamp(_pick_day(rng, month_start, rng.randint(10, 25)), as_of)
        push(
            bank=rep_acct.bank,
            email_type="account_credit_repayment",
            direction=C.DIRECTION_CREDIT,
            amount=rep,
            txn_date=d,
            counterparty="LOAN REPAYMENT",
            category="repayment",
            source="email",
            account_pk=rep_acct.pk,
            card_pk=None,
            card_mask=None,
            account_mask=_mask(rep_acct, 6),
            reference_number=txn_reference(stable_id("txn-repay", d.isoformat())),
            channel="neft",
            balance=bal,
            ledger_account=f"Assets:Bank:{_ledger_leaf(rep_acct)}",
            ledger_counterpart="Income:Repayment",
        )

        # --- Rent (debit, monthly, NACH → reference nullified) ---
        rent = money(rng, 18_000_00, 28_000_00)
        rent_acct = savings[0]
        bal = apply_balance(rent_acct.pk, C.DIRECTION_DEBIT, rent)
        d = _clamp(_pick_day(rng, month_start, 3), as_of)
        push(
            bank=rent_acct.bank,
            email_type="account_debit_nach",
            direction=C.DIRECTION_DEBIT,
            amount=rent,
            txn_date=d,
            counterparty="LANDLORD",
            category="rent",
            source="email",
            account_pk=rent_acct.pk,
            card_pk=None,
            card_mask=None,
            account_mask=_mask(rent_acct, 6),
            # NACH channels have their reference nullified by a dashboard
            # migration; emit None here to exercise that row shape.
            reference_number=None,
            channel="nach",
            balance=bal,
            ledger_account="Expenses:Rent",
            ledger_counterpart=f"Assets:Bank:{_ledger_leaf(rent_acct)}",
        )

        # --- Utilities bill (paired email + SMS → dedup/merge case) ---
        util = money(rng, 800_00, 2_500_00)
        util_acct = savings[0]
        bal = apply_balance(util_acct.pk, C.DIRECTION_DEBIT, util)
        d = _clamp(_pick_day(rng, month_start, 5), as_of)
        group = stable_id("dedup", d.isoformat(), "utility")
        ref = txn_reference(stable_id("txn-util", d.isoformat()))
        email_pk_util = add_email(
            util_acct.bank,
            datetime.datetime.combine(d, datetime.time(10, 30)),
            f"{util_acct.bank} utility debit",
        )
        sms_pk_util = add_sms(
            util_acct.bank,
            datetime.datetime.combine(d, datetime.time(10, 31)),
            f"INR {util} debited from XX{_mask(util_acct, 4)} ref {ref}",
        )
        txns.append(
            _txn(
                rng,
                len(txns),
                bank=util_acct.bank,
                email_type="account_debit_upi",
                direction=C.DIRECTION_DEBIT,
                amount=util,
                txn_date=d,
                counterparty="ELECTRICITY BOARD",
                category="utilities",
                # This event arrives as BOTH an email and an SMS that the loader
                # merges; its truthful source is therefore ``sms+email``.
                source="sms+email",
                account_pk=util_acct.pk,
                card_pk=None,
                card_mask=None,
                account_mask=_mask(util_acct, 6),
                reference_number=ref,
                channel="upi",
                balance=bal,
                email_pk=email_pk_util,
                sms_pk=sms_pk_util,
                dedup_group=group,
                ledger_account="Expenses:Utilities",
                ledger_counterpart=f"Assets:Bank:{_ledger_leaf(util_acct)}",
            )
        )

        # --- CC purchase (debit on credit-card account) ---
        cc_amt = money(rng, 500_00, 8_000_00)
        merchant, merch_cat = rng.choice(_MERCHANTS)
        d = _clamp(_pick_day(rng, month_start, 7), as_of)
        apply_cc(primary_cc.pk, C.DIRECTION_DEBIT, cc_amt)
        push(
            bank=primary_cc.bank,
            email_type="cc_debit_purchase",
            direction=C.DIRECTION_DEBIT,
            amount=cc_amt,
            txn_date=d,
            counterparty=merchant,
            category=merch_cat,
            source="email",
            account_pk=primary_cc.pk,
            card_pk=_card_for(cards, primary_cc.pk),
            card_mask=_mask(primary_cc, 4),
            account_mask=None,
            reference_number=txn_reference(
                stable_id("txn-cc", d.isoformat(), merchant)
            ),
            channel="pos",
            balance=None,
            ledger_account=f"Expenses:{merch_cat.replace('_', ' ').title()}",
            ledger_counterpart=f"Liabilities:Card:{_ledger_leaf(primary_cc)}",
        )

        # --- CC payment (debit from savings paying the CC) -----------------
        # Carries an explicit ``card_id`` (the primary card's pk) so the
        # projection's exact-match card resolution resolves it to the selected
        # card liability (``card_payments_resolved``); the orphan CC payment
        # edge below supplies the deliberately-unresolved counter-case.
        pay = money(rng, 5_000_00, 15_000_00)
        pay_acct = salary_acct
        bal = apply_balance(pay_acct.pk, C.DIRECTION_DEBIT, pay)
        d = _clamp(_pick_day(rng, month_start, 12), as_of)
        primary_card = next(
            c for c in cards if c.account_pk == primary_cc.pk and c.is_primary
        )
        push(
            bank=pay_acct.bank,
            email_type="account_debit_cc_payment",
            direction=C.DIRECTION_DEBIT,
            amount=pay,
            txn_date=d,
            counterparty=primary_cc.label,
            category="credit_card_payment",
            source="email",
            account_pk=pay_acct.pk,
            card_pk=primary_card.pk,
            card_mask=primary_card.card_mask,
            account_mask=_mask(pay_acct, 6),
            reference_number=None,  # missing-ref case (resolution is by card_id)
            channel="upi",
            balance=bal,
            ledger_account=f"Liabilities:Card:{_ledger_leaf(primary_cc)}",
            ledger_counterpart=f"Assets:Bank:{_ledger_leaf(pay_acct)}",
        )

        # --- Investment contribution (debit, monthly) ----------------------
        # A monthly SIP-style contribution sized so cumulative contributions
        # always exceed the (smaller, later) investment-redemption edge. This
        # keeps ``Assets:Investments:Unallocated`` non-negative chronologically
        # in the projection — a redemption without preceding contributions
        # would drive the Unallocated asset negative.
        invest = money(rng, 20_00_000, 40_00_000)
        invest_acct = rng.choice(savings)
        bal = apply_balance(invest_acct.pk, C.DIRECTION_DEBIT, invest)
        d = _clamp(_pick_day(rng, month_start, 6), as_of)
        push(
            bank=invest_acct.bank,
            email_type="account_debit_investment",
            direction=C.DIRECTION_DEBIT,
            amount=invest,
            txn_date=d,
            counterparty="FUND HOUSE",
            category="investment",
            source="email",
            account_pk=invest_acct.pk,
            card_pk=None,
            card_mask=None,
            account_mask=_mask(invest_acct, 6),
            reference_number=txn_reference(
                stable_id("txn-invest", d.isoformat(), str(len(txns)))
            ),
            channel="neft",
            balance=bal,
            ledger_account="Assets:Investments:Synthetic Fund",
            ledger_counterpart=f"Assets:Bank:{_ledger_leaf(invest_acct)}",
        )

        # --- Self-transfer pair (debit + credit share a reference) ---
        xfer = money(rng, 5_000_00, 25_000_00)
        src_acct = savings[0]
        dst_acct = salary_acct if salary_acct != src_acct else savings[1]
        bal_src = apply_balance(src_acct.pk, C.DIRECTION_DEBIT, xfer)
        bal_dst = apply_balance(dst_acct.pk, C.DIRECTION_CREDIT, xfer)
        d = _clamp(_pick_day(rng, month_start, 9), as_of)
        xfer_ref = txn_reference(stable_id("txn-xfer", d.isoformat()))
        push(
            bank=src_acct.bank,
            email_type="account_debit_transfer",
            direction=C.DIRECTION_DEBIT,
            amount=xfer,
            txn_date=d,
            counterparty="SELF",
            category="self_transfer",
            source="email",
            account_pk=src_acct.pk,
            card_pk=None,
            card_mask=None,
            account_mask=_mask(src_acct, 6),
            reference_number=xfer_ref,
            channel="neft",
            balance=bal_src,
            ledger_account=f"Assets:Bank:{_ledger_leaf(dst_acct)}",
            ledger_counterpart=f"Assets:Bank:{_ledger_leaf(src_acct)}",
        )
        push(
            bank=dst_acct.bank,
            email_type="account_credit_transfer",
            direction=C.DIRECTION_CREDIT,
            amount=xfer,
            txn_date=d,
            counterparty="SELF",
            category="self_transfer",
            source="email",
            account_pk=dst_acct.pk,
            card_pk=None,
            card_mask=None,
            account_mask=_mask(dst_acct, 6),
            reference_number=xfer_ref,
            channel="neft",
            balance=bal_dst,
            ledger_account=f"Assets:Bank:{_ledger_leaf(dst_acct)}",
            ledger_counterpart=f"Assets:Bank:{_ledger_leaf(src_acct)}",
        )

        # --- Cash withdrawal (debit, cash_withdrawal) ---
        wd = money(rng, 2_000_00, 10_000_00)
        wd_acct = savings[-1]
        bal = apply_balance(wd_acct.pk, C.DIRECTION_DEBIT, wd)
        d = _clamp(_pick_day(rng, month_start, 11), as_of)
        push(
            bank=wd_acct.bank,
            email_type="account_debit_atm",
            direction=C.DIRECTION_DEBIT,
            amount=wd,
            txn_date=d,
            counterparty="ATM",
            category="cash_withdrawal",
            source="email",
            account_pk=wd_acct.pk,
            card_pk=None,
            card_mask=_DEBIT_CARD_MASK,
            account_mask=_mask(wd_acct, 6),
            reference_number=txn_reference(stable_id("txn-wd", d.isoformat())),
            channel="atm",
            balance=bal,
            ledger_account="Expenses:Cash Withdrawal",
            ledger_counterpart=f"Assets:Bank:{_ledger_leaf(wd_acct)}",
        )

        # --- Fees & a refund/reversal (small structural variety) ---
        fee = money(rng, 50_00, 500_00)
        fee_acct = savings[0]
        bal = apply_balance(fee_acct.pk, C.DIRECTION_DEBIT, fee)
        d = _clamp(_pick_day(rng, month_start, 14), as_of)
        push(
            bank=fee_acct.bank,
            email_type="account_debit_charge",
            direction=C.DIRECTION_DEBIT,
            amount=fee,
            txn_date=d,
            counterparty=fee_acct.bank.upper(),
            category="fees_charges",
            source="email",
            account_pk=fee_acct.pk,
            card_pk=None,
            card_mask=None,
            account_mask=_mask(fee_acct, 6),
            reference_number=txn_reference(stable_id("txn-fee", d.isoformat())),
            channel="charge",
            balance=bal,
            ledger_account="Expenses:Fees Charges",
            ledger_counterpart=f"Assets:Bank:{_ledger_leaf(fee_acct)}",
        )

        # --- Generic volume rows to reach the profile target ---------------
        # Profile-scaled amount band so the bank-scope income/expense ratio
        # stays truthful (>= 1.1 smoke/ci, >= 1.0 stress) at every scale: a
        # stress profile emits far more rows but each at a far smaller amount.
        # ~12% of the rows are small credits (refunds / cashback / occasional
        # freelance income) — the realistic small-credit tail that raises the
        # credit share and, for contra-expense slugs, trims net spend.
        for _ in range(prof.monthly_volume):
            merchant, merch_cat = rng.choice(_MERCHANTS)
            is_credit = rng.random() < 0.12
            acct = rng.choice(savings)
            d = _clamp(_pick_day(rng, month_start, rng.randint(1, 27)), as_of)
            idx = str(len(txns))
            if is_credit:
                # Small refund / cashback (contra-expense) most of the time;
                # occasionally a small freelance credit (income).
                if rng.random() < 0.2:
                    cat = "other_income"
                    email_type = "account_credit_income"
                    counterparty = "FREELANCE CLIENT"
                    direction = C.DIRECTION_CREDIT
                    ledger_account = f"Assets:Bank:{_ledger_leaf(acct)}"
                    ledger_counterpart = "Income:Other Income"
                    channel = "neft"
                else:
                    cat = rng.choice(("refund", "cashback_rewards"))
                    email_type = "account_credit_refund"
                    counterparty = merchant
                    direction = C.DIRECTION_CREDIT
                    # Contra-expense: posts against the merchant's expense bucket.
                    ledger_account = f"Expenses:{merch_cat.replace('_', ' ').title()}"
                    ledger_counterpart = f"Assets:Bank:{_ledger_leaf(acct)}"
                    channel = "upi"
                amt = money(rng, 50, prof.expense_high_paise)
                bal = apply_balance(acct.pk, C.DIRECTION_CREDIT, amt)
                ref = txn_reference(
                    stable_id("txn-gen-cr", d.isoformat(), str(amt), merchant, idx)
                )
            else:
                email_type = "account_debit_upi"
                counterparty = merchant
                cat = merch_cat
                direction = C.DIRECTION_DEBIT
                ledger_account = f"Expenses:{merch_cat.replace('_', ' ').title()}"
                ledger_counterpart = f"Assets:Bank:{_ledger_leaf(acct)}"
                channel = "upi"
                amt = money(rng, prof.expense_low_paise, prof.expense_high_paise)
                bal = apply_balance(acct.pk, C.DIRECTION_DEBIT, amt)
                ref = txn_reference(
                    stable_id("txn-gen", d.isoformat(), str(amt), merchant, idx)
                )
            push(
                bank=acct.bank,
                email_type=email_type,
                direction=direction,
                amount=amt,
                txn_date=d,
                counterparty=counterparty,
                category=cat,
                source="email",
                account_pk=acct.pk,
                card_pk=None,
                card_mask=None,
                account_mask=_mask(acct, 6),
                # Fold in the per-call transaction index so each generic row
                # gets a unique reference. Without it, two rows that happen to
                # share (day, amount, merchant) collide on the dashboard's
                # unique (bank, reference_number, direction) index, and the
                # bulk lane's ON CONFLICT DO NOTHING silently drops one — a loss
                # that only surfaces at stress scale (>=200k rows).
                reference_number=ref,
                channel=channel,
                balance=bal,
                txn_time=datetime.time(rng.randint(8, 21), rng.randint(0, 59)),
                ledger_account=ledger_account,
                ledger_counterpart=ledger_counterpart,
            )

    # --- Cross-cutting edge cases (a small, fixed set) --------------------
    _add_edge_cases(
        rng,
        as_of,
        accounts,
        cards,
        savings,
        primary_cc,
        push,
        apply_balance,
        apply_cc,
    )

    # --- Full category-vocabulary coverage (one row per otherwise-missing slug)
    _add_category_coverage(rng, as_of, savings, primary_cc, txns, push, apply_balance)

    return txns, emails, sms, balances, cc_outstanding


def _txn(
    rng: random.Random,
    index: int,
    *,
    bank: str,
    email_type: str,
    direction: str,
    amount: Decimal,
    txn_date: datetime.date,
    counterparty: str | None,
    category: str | None,
    source: str,
    account_pk: int | None,
    card_pk: int | None,
    card_mask: str | None,
    account_mask: str | None,
    reference_number: str | None,
    channel: str | None,
    balance: Decimal | None,
    email_pk: int | None,
    sms_pk: int | None,
    ledger_account: str,
    ledger_counterpart: str,
    dedup_group: str | None = None,
    currency: str = "INR",
    review_status: str | None = None,
    category_method: str | None = None,
    category_confidence: float | None = None,
    category_model: str | None = None,
    review_reason: str | None = None,
) -> SynthTransaction:
    sid = stable_id("txn", str(index), bank, str(amount), txn_date.isoformat())
    return SynthTransaction(
        stable_id=sid,
        bank=bank,
        email_type=email_type,
        direction=direction,
        amount=amount,
        currency=currency,
        transaction_date=txn_date,
        transaction_time=None,
        counterparty=counterparty,
        card_mask=card_mask,
        account_mask=account_mask,
        reference_number=reference_number,
        channel=channel,
        balance=balance,
        raw_description=f"synthetic {email_type}",
        category=category,
        source=source,
        account_pk=account_pk,
        card_pk=card_pk,
        email_pk=email_pk,
        sms_pk=sms_pk,
        dedup_group=dedup_group,
        ledger_account=ledger_account,
        ledger_counterpart=ledger_counterpart,
        review_status=review_status,
        category_method=category_method,
        category_confidence=category_confidence,
        category_model=category_model,
        review_reason=review_reason,
    )


def _add_edge_cases(
    rng: random.Random,
    as_of: datetime.date,
    accounts: list[SynthAccount],
    cards: list[SynthCard],
    savings: list[SynthAccount],
    primary_cc: SynthAccount,
    push,
    apply_balance,
    apply_cc,
) -> None:
    # Non-INR transactions (see _build_fx_rates). The scenario's FX map covers
    # USD and EUR but not GBP, and the earliest USD rate is ~190 days before
    # as_of, so these four rows split cleanly into covered and missing-rate
    # cases the projection's ``priced`` policy exercises:
    #
    # * USD on as_of-3   → covered (latest USD rate on/before is as_of-5)
    # * EUR on as_of-7   → covered (EUR rate as_of-192)
    # * GBP on as_of-4   → missing_fx_rate (no GBP rate configured)
    # * USD on as_of-250 → missing_fx_rate (before the first USD rate as_of-192),
    #   dated AFTER the backdated INR row (as_of-400) so it does not shift the
    #   corpus opening-balance cutover (the earliest transaction date).
    acct = savings[0]

    def _foreign(
        *,
        days_back: int,
        currency: str,
        amount: str,
        counterparty: str,
        category: str,
        balance_drain: str | None,
    ) -> None:
        d = as_of - datetime.timedelta(days=days_back)
        bal = (
            apply_balance(acct.pk, C.DIRECTION_DEBIT, Decimal(balance_drain))
            if balance_drain
            else None
        )
        push(
            bank=acct.bank,
            email_type="account_debit_international",
            direction=C.DIRECTION_DEBIT,
            amount=quantize(Decimal(amount)),
            txn_date=d,
            counterparty=counterparty,
            category=category,
            source="email",
            account_pk=acct.pk,
            card_pk=None,
            card_mask=None,
            account_mask=_mask(acct, 6),
            reference_number=txn_reference(
                stable_id("txn-fx", currency, d.isoformat())
            ),
            channel="international",
            balance=bal,
            currency=currency,
            ledger_account=f"Expenses:{category.replace('_', ' ').title()}",
            ledger_counterpart=f"Assets:Bank:{_ledger_leaf(acct)}",
        )

    _foreign(
        days_back=3,
        currency="USD",
        amount="42.99",
        counterparty="FOREIGN MERCHANT USD",
        category="shopping",
        balance_drain="3200.00",
    )
    _foreign(
        days_back=7,
        currency="EUR",
        amount="60.00",
        counterparty="EU SOFTWARE SRL",
        category="subscriptions",
        balance_drain="5600.00",
    )
    _foreign(
        days_back=4,
        currency="GBP",
        amount="25.00",
        counterparty="LONDON STORES LTD",
        category="shopping",
        balance_drain="2900.00",
    )
    _foreign(
        days_back=250,
        currency="USD",
        amount="12.50",
        counterparty="LEGACY USD CHARGE",
        category="misc",
        balance_drain=None,
    )

    # Backdated row (well before the normal window).
    d = (as_of - datetime.timedelta(days=400)).replace(day=2)
    push(
        bank=acct.bank,
        email_type="account_debit_upi",
        direction=C.DIRECTION_DEBIT,
        amount=quantize(Decimal("199.00")),
        txn_date=d,
        counterparty="BACKDATED MERCHANT",
        category="misc",
        source="email",
        account_pk=acct.pk,
        card_pk=None,
        card_mask=None,
        account_mask=_mask(acct, 6),
        reference_number=txn_reference(stable_id("txn-back", d.isoformat())),
        channel="upi",
        balance=None,
        ledger_account="Expenses:Misc",
        ledger_counterpart=f"Assets:Bank:{_ledger_leaf(acct)}",
    )

    # Long, spaced account name (>=47 chars) — exercises the renderer's
    # amount-separation guard: ledger needs >=2 spaces between the account
    # name and the amount, and a name this long overruns the alignment
    # column. The spaces in the leaf are retained verbatim (not collapsed)
    # so the exact account name survives into the journal.
    d = as_of - datetime.timedelta(days=6)
    push(
        bank=acct.bank,
        email_type="account_debit_nach",
        direction=C.DIRECTION_DEBIT,
        amount=quantize(Decimal("18499.00")),
        txn_date=d,
        counterparty="INSURER CO",
        category="insurance",
        source="email",
        account_pk=acct.pk,
        card_pk=None,
        card_mask=None,
        account_mask=_mask(acct, 6),
        reference_number=txn_reference(stable_id("txn-ins", d.isoformat())),
        channel="nach",
        balance=None,
        ledger_account="Expenses:Insurance:Medical Health Insurance Premium",
        ledger_counterpart=f"Assets:Bank:{_ledger_leaf(acct)}",
    )

    # Unlinked unknown (no account/card, no mask, unknown category).
    d = as_of - datetime.timedelta(days=2)
    push(
        bank="unknown",
        email_type="account_debit_upi",
        direction=C.DIRECTION_DEBIT,
        amount=quantize(Decimal("777.00")),
        txn_date=d,
        counterparty="UNKNOWN MERCHANT",
        category="unknown",
        source="email",
        account_pk=None,
        card_pk=None,
        card_mask=None,
        account_mask=None,
        reference_number=None,
        channel="upi",
        balance=None,
        ledger_account="Expenses:Unknown",
        ledger_counterpart="Equity:Unknown",
        # An uncategorized, unlinked row is exactly what the manual-review queue
        # flags for a human — seed it as ``flagged`` so that surface is exercised.
        review_status="flagged",
    )

    # --- Refund: a merchant credit returning money to the bank. ------------
    acct = savings[0]
    d = as_of - datetime.timedelta(days=13)
    bal = apply_balance(acct.pk, C.DIRECTION_CREDIT, quantize(Decimal("1499.00")))
    push(
        bank=acct.bank,
        email_type="account_credit_refund",
        direction=C.DIRECTION_CREDIT,
        amount=quantize(Decimal("1499.00")),
        txn_date=d,
        counterparty="MERCHANT REFUND",
        category="refund",
        source="email",
        account_pk=acct.pk,
        card_pk=None,
        card_mask=None,
        account_mask=_mask(acct, 6),
        reference_number=txn_reference(stable_id("txn-refund", d.isoformat())),
        channel="neft",
        balance=bal,
        ledger_account="Income:Refund",
        ledger_counterpart=f"Assets:Bank:{_ledger_leaf(acct)}",
    )

    # --- Reversal: a bank credit reversing a prior debit (a fee reversal). -
    d = as_of - datetime.timedelta(days=16)
    bal = apply_balance(acct.pk, C.DIRECTION_CREDIT, quantize(Decimal("236.00")))
    push(
        bank=acct.bank,
        email_type="account_credit_reversal",
        direction=C.DIRECTION_CREDIT,
        amount=quantize(Decimal("236.00")),
        txn_date=d,
        counterparty="BANK REVERSAL",
        # Same category as the charge it reverses so the cashflow report nets
        # them out in the same bucket.
        category="fees_charges",
        source="email",
        account_pk=acct.pk,
        card_pk=None,
        card_mask=None,
        account_mask=_mask(acct, 6),
        reference_number=txn_reference(stable_id("txn-reversal", d.isoformat())),
        channel="neft",
        balance=bal,
        ledger_account="Expenses:Fees Charges",
        ledger_counterpart=f"Assets:Bank:{_ledger_leaf(acct)}",
    )

    # --- Unmatched self-transfer: a single debit leg with a unique reference
    # and NO matching credit. The projection's pairing logic reports this as
    # ``unmatched_self_transfer`` (a 1-debit group), exercising that skip path.
    src_acct = savings[0]
    dst_acct = savings[-1]
    d = as_of - datetime.timedelta(days=18)
    xfer_amt = money(rng, 3_000_00, 9_000_00)
    bal = apply_balance(src_acct.pk, C.DIRECTION_DEBIT, xfer_amt)
    push(
        bank=src_acct.bank,
        email_type="account_debit_transfer",
        direction=C.DIRECTION_DEBIT,
        amount=xfer_amt,
        txn_date=d,
        counterparty="SELF",
        category="self_transfer",
        source="email",
        account_pk=src_acct.pk,
        card_pk=None,
        card_mask=None,
        account_mask=_mask(src_acct, 6),
        reference_number=txn_reference(stable_id("txn-xfer-unmatched", d.isoformat())),
        channel="neft",
        balance=bal,
        ledger_account=f"Assets:Bank:{_ledger_leaf(dst_acct)}",
        ledger_counterpart=f"Assets:Bank:{_ledger_leaf(src_acct)}",
    )

    # --- Card-payment CARD side: the credit leg of a CC bill payment landing
    # on the credit-card (liability) account. The projection skips it as
    # ``card_side_payment`` (the bank-side debit is the authoritative leg).
    # The bank-side leg is already produced monthly above.
    d = as_of - datetime.timedelta(days=12)
    pay_amt = money(rng, 5_000_00, 12_000_00)
    apply_cc(primary_cc.pk, C.DIRECTION_CREDIT, pay_amt)
    push(
        bank=primary_cc.bank,
        email_type="account_credit_card_payment",
        direction=C.DIRECTION_CREDIT,
        amount=pay_amt,
        txn_date=d,
        counterparty="PAYMENT THANK YOU",
        category="credit_card_payment",
        source="email",
        account_pk=primary_cc.pk,
        card_pk=_card_for(cards, primary_cc.pk),
        card_mask=_mask(primary_cc, 4),
        account_mask=None,
        reference_number=txn_reference(stable_id("txn-ccpay-cardside", d.isoformat())),
        channel="neft",
        balance=None,
        ledger_account=f"Liabilities:Card:{_ledger_leaf(primary_cc)}",
        ledger_counterpart=f"Assets:Bank:{_ledger_leaf(acct)}",
    )

    # =====================================================================
    # Canonical scenario-branch expansion (1.3.0). Each block is a distinct,
    # documented edge the corpus exists to exercise; the coverage registry
    # detects them by their distinguishing features.
    # =====================================================================

    # --- CC purchase inside the statement window --------------------------
    # A CC debit dated within the CC statement window so the reconciliation
    # surface (and the real ``reconcile_statement`` offline pass) sees a CC
    # row that exact-matches a DB transaction. Without it the primary card has
    # no in-window debits and the CC reconciliation counts stay zero.
    cc_window_d = as_of - datetime.timedelta(days=30)
    cc_win_amt = quantize(Decimal("1234.00"))
    apply_cc(primary_cc.pk, C.DIRECTION_DEBIT, cc_win_amt)
    push(
        bank=primary_cc.bank,
        email_type="cc_debit_purchase",
        direction=C.DIRECTION_DEBIT,
        amount=cc_win_amt,
        txn_date=cc_window_d,
        counterparty="WINDOW MERCHANT",
        category="shopping",
        source="email",
        account_pk=primary_cc.pk,
        card_pk=_card_for(cards, primary_cc.pk),
        card_mask=_mask(primary_cc, 4),
        account_mask=None,
        reference_number=txn_reference(
            stable_id("txn-cc-window", cc_window_d.isoformat())
        ),
        channel="pos",
        balance=None,
        ledger_account="Expenses:Shopping",
        ledger_counterpart=f"Liabilities:Card:{_ledger_leaf(primary_cc)}",
    )

    # --- CC refund / reversal credit (NOT a bill payment) -----------------
    # A merchant refund landing on the credit-card (liability) account — it
    # reduces outstanding, but its category is ``refund`` (never
    # ``credit_card_payment``) so the projection treats it as a contra-expense,
    # not a card-side bill payment. Distinct from the bank-side merchant refund.
    d = as_of - datetime.timedelta(days=8)
    cc_refund = quantize(Decimal("750.00"))
    apply_cc(primary_cc.pk, C.DIRECTION_CREDIT, cc_refund)
    push(
        bank=primary_cc.bank,
        email_type="cc_credit_refund",
        direction=C.DIRECTION_CREDIT,
        amount=cc_refund,
        txn_date=d,
        counterparty="CC MERCHANT REFUND",
        category="refund",
        source="email",
        account_pk=primary_cc.pk,
        card_pk=_card_for(cards, primary_cc.pk),
        card_mask=_mask(primary_cc, 4),
        account_mask=None,
        reference_number=txn_reference(stable_id("txn-cc-refund", d.isoformat())),
        channel="refund",
        balance=None,
        ledger_account="Expenses:Shopping",
        ledger_counterpart=f"Liabilities:Card:{_ledger_leaf(primary_cc)}",
    )

    # --- Investment redemption (credit) ---------------------------------
    # A portfolio redemption returning money to the bank — the income-style
    # credit that nets against investment contributions in the cashflow
    # investment bucket. Sized smaller than a single monthly contribution and
    # dated after them, so cumulative contributions always exceed redemptions
    # and ``Assets:Investments:Unallocated`` never goes negative chronologically.
    acct = savings[0]
    d = as_of - datetime.timedelta(days=4)
    redeem = quantize(Decimal("15000.00"))
    bal = apply_balance(acct.pk, C.DIRECTION_CREDIT, redeem)
    push(
        bank=acct.bank,
        email_type="account_credit_redemption",
        direction=C.DIRECTION_CREDIT,
        amount=redeem,
        txn_date=d,
        counterparty="FUND HOUSE REDEMPTION",
        category="investment_redemption",
        source="email",
        account_pk=acct.pk,
        card_pk=None,
        card_mask=None,
        account_mask=_mask(acct, 6),
        reference_number=txn_reference(stable_id("txn-redeem", d.isoformat())),
        channel="neft",
        balance=bal,
        ledger_account="Assets:Investments:Synthetic Fund",
        ledger_counterpart=f"Assets:Bank:{_ledger_leaf(acct)}",
    )

    # --- Invalid currency edge (XXX) + blank currency edge -----------------
    # ``XXX`` is an unmapped currency code (non-INR, never priced) — exercises
    # the non-INR exclusion paths without inventing a real currency. The blank
    # (None) currency is read as INR by the report; both keep balance=None so
    # they do not perturb the tracked running-balance invariant.
    acct = savings[0]
    d = as_of - datetime.timedelta(days=9)
    push(
        bank=acct.bank,
        email_type="account_debit_international",
        direction=C.DIRECTION_DEBIT,
        amount=quantize(Decimal("99.00")),
        txn_date=d,
        counterparty="MALFORMED CURRENCY MERCHANT",
        category="misc",
        source="email",
        account_pk=acct.pk,
        card_pk=None,
        card_mask=None,
        account_mask=_mask(acct, 6),
        reference_number=txn_reference(stable_id("txn-invcurrency", d.isoformat())),
        channel="international",
        balance=None,
        # A genuinely projection-invalid currency token: the projection's
        # sanitizer rejects a cleaned value that does not start with a letter,
        # so ``000`` → ``invalid_currency`` (not ``missing_fx_rate`` like the
        # ISO placeholder ``XXX``). Represents a parser emitting a numeric value
        # into the currency field. balance=None so it does not perturb the
        # tracked running-balance invariant (the projection skips it anyway).
        currency=C.INVALID_CURRENCY_TOKEN,
        ledger_account="Expenses:Misc",
        ledger_counterpart=f"Assets:Bank:{_ledger_leaf(acct)}",
    )
    # Blank currency (None) — a legacy row that predates the currency column's
    # default. The report treats NULL currency as INR; keep it a tiny expense so
    # it lands in the uncategorized/expense population without distorting ratio.
    # apply_balance is called (with balance=None on the row) so the tracked
    # running balance stays consistent with the projection's INR running
    # balance, which emits this row as an INR posting.
    d = as_of - datetime.timedelta(days=10)
    apply_balance(acct.pk, C.DIRECTION_DEBIT, quantize(Decimal("12.00")))
    push(
        bank=acct.bank,
        email_type="account_debit_upi",
        direction=C.DIRECTION_DEBIT,
        amount=quantize(Decimal("12.00")),
        txn_date=d,
        counterparty="LEGACY NO-CURRENCY ROW",
        category="misc",
        source="email",
        account_pk=acct.pk,
        card_pk=None,
        card_mask=None,
        account_mask=_mask(acct, 6),
        reference_number=txn_reference(stable_id("txn-nocurr", d.isoformat())),
        channel="upi",
        balance=None,
        currency=None,
        ledger_account="Expenses:Misc",
        ledger_counterpart=f"Assets:Bank:{_ledger_leaf(acct)}",
    )

    # --- Undated transaction ----------------------------------------------
    # A row with ``transaction_date is None`` — a legitimate shape the cashflow
    # report's Undated footnote counts. Its source email still carries a
    # received_at (noon on as_of) so the loader can build the row.
    acct = savings[0]
    push(
        bank=acct.bank,
        email_type="account_debit_upi",
        direction=C.DIRECTION_DEBIT,
        amount=quantize(Decimal("333.00")),
        txn_date=None,
        counterparty="UNDATED MERCHANT",
        category="misc",
        source="email",
        account_pk=acct.pk,
        card_pk=None,
        card_mask=None,
        account_mask=_mask(acct, 6),
        reference_number=txn_reference(stable_id("txn-undated", "1")),
        channel="upi",
        balance=None,
        ledger_account="Expenses:Misc",
        ledger_counterpart=f"Assets:Bank:{_ledger_leaf(acct)}",
    )

    # --- Blank counterparty + blank category ------------------------------
    # A bank-side row with a blank counterparty and a separate row with a blank
    # category — exercises the report's blank-normalization populations.
    acct = savings[0]
    d = as_of - datetime.timedelta(days=11)
    bal = apply_balance(acct.pk, C.DIRECTION_DEBIT, quantize(Decimal("88.00")))
    push(
        bank=acct.bank,
        email_type="account_debit_upi",
        direction=C.DIRECTION_DEBIT,
        amount=quantize(Decimal("88.00")),
        txn_date=d,
        counterparty=None,
        category="misc",
        source="email",
        account_pk=acct.pk,
        card_pk=None,
        card_mask=None,
        account_mask=_mask(acct, 6),
        reference_number=txn_reference(stable_id("txn-nocp", d.isoformat())),
        channel="upi",
        balance=bal,
        ledger_account="Expenses:Misc",
        ledger_counterpart=f"Assets:Bank:{_ledger_leaf(acct)}",
    )
    d = as_of - datetime.timedelta(days=19)
    bal = apply_balance(acct.pk, C.DIRECTION_DEBIT, quantize(Decimal("77.00")))
    push(
        bank=acct.bank,
        email_type="account_debit_upi",
        direction=C.DIRECTION_DEBIT,
        amount=quantize(Decimal("77.00")),
        txn_date=d,
        counterparty="UNCATEGORIZED MERCHANT",
        category=None,
        source="email",
        account_pk=acct.pk,
        card_pk=None,
        card_mask=None,
        account_mask=_mask(acct, 6),
        reference_number=txn_reference(stable_id("txn-nocat", d.isoformat())),
        channel="upi",
        balance=bal,
        ledger_account="Expenses:Unknown",
        ledger_counterpart=f"Assets:Bank:{_ledger_leaf(acct)}",
    )

    # --- AM/PM same-day pair ----------------------------------------------
    # Two debits on the same day, one AM and one PM, sharing an AMPM reference
    # prefix so coverage can detect the pair shape. Distinct references so they
    # do not collapse on the natural-key index.
    acct = savings[0]
    d = as_of - datetime.timedelta(days=14)
    am_amt = quantize(Decimal("150.00"))
    pm_amt = quantize(Decimal("250.00"))
    bal = apply_balance(acct.pk, C.DIRECTION_DEBIT, am_amt + pm_amt)
    push(
        bank=acct.bank,
        email_type="account_debit_upi",
        direction=C.DIRECTION_DEBIT,
        amount=am_amt,
        txn_date=d,
        txn_time=datetime.time(10, 15),
        counterparty="COFFEE DAY",
        category="dining",
        source="email",
        account_pk=acct.pk,
        card_pk=None,
        card_mask=None,
        account_mask=_mask(acct, 6),
        reference_number=f"SYN-AMPM-AM-{d.isoformat()}",
        channel="upi",
        balance=None,
        ledger_account="Expenses:Dining",
        ledger_counterpart=f"Assets:Bank:{_ledger_leaf(acct)}",
    )
    push(
        bank=acct.bank,
        email_type="account_debit_upi",
        direction=C.DIRECTION_DEBIT,
        amount=pm_amt,
        txn_date=d,
        txn_time=datetime.time(20, 45),
        counterparty="DINNER RESTAURANT",
        category="dining",
        source="email",
        account_pk=acct.pk,
        card_pk=None,
        card_mask=None,
        account_mask=_mask(acct, 6),
        reference_number=f"SYN-AMPM-PM-{d.isoformat()}",
        channel="upi",
        balance=bal,
        ledger_account="Expenses:Dining",
        ledger_counterpart=f"Assets:Bank:{_ledger_leaf(acct)}",
    )

    # --- Ref-mismatch shape ------------------------------------------------
    # A single bank-side row carrying a deliberately mismatched reference
    # (``SYN-MISMATCH`` prefix) so the reconcile surface can exercise the
    # fuzzy-date refusal path when a statement row's ref disagrees with the
    # DB ref. Real, linkable transaction — the mismatch is a reconcile concern.
    acct = savings[0]
    d = as_of - datetime.timedelta(days=22)
    bal = apply_balance(acct.pk, C.DIRECTION_DEBIT, quantize(Decimal("640.00")))
    push(
        bank=acct.bank,
        email_type="account_debit_upi",
        direction=C.DIRECTION_DEBIT,
        amount=quantize(Decimal("640.00")),
        txn_date=d,
        counterparty="MISMATCH MERCHANT",
        category="shopping",
        source="email",
        account_pk=acct.pk,
        card_pk=None,
        card_mask=None,
        account_mask=_mask(acct, 6),
        reference_number=f"SYN-MISMATCH-{d.isoformat()}",
        channel="upi",
        balance=bal,
        ledger_account="Expenses:Shopping",
        ledger_counterpart=f"Assets:Bank:{_ledger_leaf(acct)}",
    )

    # --- Balance-conflict split -------------------------------------------
    # Two legs of a split where the stated running balance disagrees (a
    # balance-conflict diagnostic shape). ``SYN-CONFLICT`` prefix for coverage.
    acct = savings[0]
    d = as_of - datetime.timedelta(days=24)
    bal = apply_balance(acct.pk, C.DIRECTION_DEBIT, quantize(Decimal("500.00")))
    push(
        bank=acct.bank,
        email_type="account_debit_upi",
        direction=C.DIRECTION_DEBIT,
        amount=quantize(Decimal("300.00")),
        txn_date=d,
        counterparty="SPLIT MERCHANT A",
        category="shopping",
        source="email",
        account_pk=acct.pk,
        card_pk=None,
        card_mask=None,
        account_mask=_mask(acct, 6),
        reference_number=f"SYN-CONFLICT-A-{d.isoformat()}",
        channel="upi",
        balance=bal,
        ledger_account="Expenses:Shopping",
        ledger_counterpart=f"Assets:Bank:{_ledger_leaf(acct)}",
    )
    push(
        bank=acct.bank,
        email_type="account_debit_upi",
        direction=C.DIRECTION_DEBIT,
        amount=quantize(Decimal("200.00")),
        txn_date=d,
        counterparty="SPLIT MERCHANT B",
        category="shopping",
        source="email",
        account_pk=acct.pk,
        card_pk=None,
        card_mask=None,
        account_mask=_mask(acct, 6),
        reference_number=f"SYN-CONFLICT-B-{d.isoformat()}",
        channel="upi",
        # Deliberately the same balance as the A leg → the conflict the
        # diagnostic surfaces (two same-day debits claiming the same balance).
        balance=bal,
        ledger_account="Expenses:Shopping",
        ledger_counterpart=f"Assets:Bank:{_ledger_leaf(acct)}",
    )

    # --- Orphan CC payment (no card-side credit) --------------------------
    # A bank-side CC bill payment whose card-side credit never arrived — an
    # orphan debit. ``SYN-ORPHAN-CCPAY`` prefix for coverage. It does NOT call
    # apply_cc (the card side is absent, which is the point).
    acct = savings[0]
    d = as_of - datetime.timedelta(days=26)
    bal = apply_balance(acct.pk, C.DIRECTION_DEBIT, quantize(Decimal("4000.00")))
    push(
        bank=acct.bank,
        email_type="account_debit_cc_payment",
        direction=C.DIRECTION_DEBIT,
        amount=quantize(Decimal("4000.00")),
        txn_date=d,
        counterparty=primary_cc.label,
        category="credit_card_payment",
        source="email",
        account_pk=acct.pk,
        card_pk=None,
        card_mask=None,
        account_mask=_mask(acct, 6),
        reference_number=f"SYN-ORPHAN-CCPAY-{d.isoformat()}",
        channel="upi",
        balance=bal,
        ledger_account=f"Liabilities:Card:{_ledger_leaf(primary_cc)}",
        ledger_counterpart=f"Assets:Bank:{_ledger_leaf(acct)}",
    )

    # --- Rich categorization metadata (method + review axis) --------------
    # A small set of rows carrying explicit category_method and review_status
    # so the metadata axis (manual/rule/llm/pending_llm) and the review queue
    # (pending/notified/resolved) are populated. These are ordinary debits; the
    # metadata is what distinguishes them.
    acct = savings[0]
    d = as_of - datetime.timedelta(days=27)
    bal = apply_balance(acct.pk, C.DIRECTION_DEBIT, quantize(Decimal("420.00")))
    push(
        bank=acct.bank,
        email_type="account_debit_upi",
        direction=C.DIRECTION_DEBIT,
        amount=quantize(Decimal("420.00")),
        txn_date=d,
        counterparty="MANUAL CAT MERCHANT",
        category="shopping",
        source="email",
        account_pk=acct.pk,
        card_pk=None,
        card_mask=None,
        account_mask=_mask(acct, 6),
        reference_number=txn_reference(stable_id("txn-manual", d.isoformat())),
        channel="upi",
        balance=bal,
        ledger_account="Expenses:Shopping",
        ledger_counterpart=f"Assets:Bank:{_ledger_leaf(acct)}",
        category_method="manual",
        review_status="resolved",
        review_reason="confirmed grocery purchase",
    )
    d = as_of - datetime.timedelta(days=28)
    bal = apply_balance(acct.pk, C.DIRECTION_DEBIT, quantize(Decimal("610.00")))
    push(
        bank=acct.bank,
        email_type="account_debit_upi",
        direction=C.DIRECTION_DEBIT,
        amount=quantize(Decimal("610.00")),
        txn_date=d,
        counterparty="LLM CAT MERCHANT",
        category="travel",
        source="email",
        account_pk=acct.pk,
        card_pk=None,
        card_mask=None,
        account_mask=_mask(acct, 6),
        reference_number=txn_reference(stable_id("txn-llm", d.isoformat())),
        channel="upi",
        balance=bal,
        ledger_account="Expenses:Travel",
        ledger_counterpart=f"Assets:Bank:{_ledger_leaf(acct)}",
        category_method="llm",
        category_confidence=0.92,
        category_model="synth-llm-v1",
        review_status="notified",
    )
    d = as_of - datetime.timedelta(days=29)
    bal = apply_balance(acct.pk, C.DIRECTION_DEBIT, quantize(Decimal("130.00")))
    push(
        bank=acct.bank,
        email_type="account_debit_upi",
        direction=C.DIRECTION_DEBIT,
        amount=quantize(Decimal("130.00")),
        txn_date=d,
        counterparty="PENDING LLM MERCHANT",
        category="misc",
        source="email",
        account_pk=acct.pk,
        card_pk=None,
        card_mask=None,
        account_mask=_mask(acct, 6),
        reference_number=txn_reference(stable_id("txn-pendingllm", d.isoformat())),
        channel="upi",
        balance=bal,
        ledger_account="Expenses:Misc",
        ledger_counterpart=f"Assets:Bank:{_ledger_leaf(acct)}",
        category_method="pending_llm",
        review_status="pending",
    )


# ---------------------------------------------------------------------------
# Category-vocabulary coverage
# ---------------------------------------------------------------------------

# Seed categories that represent money IN (a credit to the bank). Everything
# else in the vocabulary is a debit (expense, asset purchase, liability paydown).
# ``repayment`` is the transfers-in slug (money handed back *to* the account
# holder), so it is a credit — emitting it as a debit would be a directionally
# impossible row the categorization polarity guard flips to ``expense``, which
# would silently mis-label the transfers-in line.
_CREDIT_CATEGORIES = frozenset(
    {
        "interest",
        "other_income",
        "cashback_rewards",
        "refund",
        "investment_redemption",
        "repayment",
    }
)

# Categories whose ledger root is not a plain ``Expenses:<Title>`` /
# ``Income:<Title>`` derivation. Investment contributions/redemptions move the
# dedicated investment asset; a repayment pays down a liability.
_CATEGORY_LEDGER_ROOT: dict[str, str] = {
    "investment": "Assets:Investments:Synthetic Fund",
    "investment_redemption": "Assets:Investments:Synthetic Fund",
    "repayment": "Liabilities:Loan:Synthetic",
}

# A realistic counterparty per category so the seeded rows are distinguishable.
_CATEGORY_COUNTERPARTY: dict[str, str] = {
    "interest": "BANK INTEREST",
    "other_income": "FREELANCE CLIENT",
    "cashback_rewards": "REWARDS PORTAL",
    "refund": "MERCHANT REFUND",
    "investment_redemption": "FUND HOUSE REDEMPTION",
    "bill_payment": "ELECTRICITY BOARD",
    "car_maintenance": "CAR SERVICE CENTER",
    "charity_gift": "GIVE CHARITY",
    "education": "TUITION ACADEMY",
    "emi_loan": "LOAN DEPARTMENT",
    "expense": "GENERAL STORES",
    "gift": "GIFT SHOP",
    "investment": "FUND HOUSE",
    "personal_care": "SALON & SPA",
    "repayment": "LOAN REPAYMENT",
    "tax": "INCOME TAX DEPT",
}


def _add_category_coverage(
    rng: random.Random,
    as_of: datetime.date,
    savings: list[SynthAccount],
    primary_cc: SynthAccount,
    txns: list[SynthTransaction],
    push,
    apply_balance,
) -> None:
    """Emit one representative transaction for every seed category the monthly
    volume + edge cases did not already produce.

    Guarantees the full :data:`constants.SEED_CATEGORY_SLUGS` vocabulary is
    exercised at every profile scale (so the cashflow buckets, the projection
    contra-accounts, and the category coverage matrix all see every slug). The
    pass is deterministic, small (one row per missing slug), and independent of
    the volume loop's RNG stream. Each row carries a unique reference (folded
    with ``len(txns)``) so it cannot collide on the dashboard's natural-key
    unique index.
    """
    present = {t.category for t in txns}
    missing = [s for s in C.SEED_CATEGORY_SLUGS if s not in present]
    if not missing:
        return

    for slug in missing:
        credit = slug in _CREDIT_CATEGORIES
        direction = C.DIRECTION_CREDIT if credit else C.DIRECTION_DEBIT
        # The CC account only hosts the card-side payment (handled above); every
        # remaining category lives on a bank account.
        acct = rng.choice(savings)
        amt = money(rng, 100_00, 6_000_00)
        bal = apply_balance(acct.pk, direction, amt)
        d = as_of - datetime.timedelta(days=rng.randint(5, 60))
        root = _CATEGORY_LEDGER_ROOT.get(slug)
        if root is None:
            prefix = "Income" if credit else "Expenses"
            root = f"{prefix}:{slug.replace('_', ' ').title()}"
        push(
            bank=acct.bank,
            email_type=f"account_{'credit' if credit else 'debit'}_{slug}",
            direction=direction,
            amount=amt,
            txn_date=d,
            counterparty=_CATEGORY_COUNTERPARTY.get(
                slug, slug.replace("_", " ").upper()
            ),
            category=slug,
            source="email",
            account_pk=acct.pk,
            card_pk=None,
            card_mask=None,
            account_mask=_mask(acct, 6),
            reference_number=txn_reference(
                stable_id("txn-catcov", slug, str(len(txns)))
            ),
            channel="upi",
            balance=bal,
            ledger_account=root,
            ledger_counterpart=f"Assets:Bank:{_ledger_leaf(acct)}",
        )


def _clamp(d: datetime.date, as_of: datetime.date) -> datetime.date:
    """Never emit a transaction dated after ``as_of`` — the cut-off is the
    scenario's truth horizon, and a future-dated row would distort every
    range-bounded figure (cashflow, trend) and break the as_of invariant."""
    return d if d <= as_of else as_of


def _pick_day(
    rng: random.Random, month_start: datetime.date, preferred: int
) -> datetime.date:
    """A deterministic day in ``month_start``'s month, biasing toward
    ``preferred`` but jittered so same-bank same-day collisions are rare.

    Callers wrap the result in :func:`_clamp` when ``month_start`` may be the
    ``as_of`` month (whose valid days stop at ``as_of``, not the month end)."""
    last = monthrange(month_start.year, month_start.month)[1]
    day = min(max(preferred + rng.randint(-2, 2), 1), last)
    return month_start.replace(day=day)


def _mask(account: SynthAccount, last: int = 6) -> str | None:
    """Last-``last`` digits of an account number, or None if it has none."""
    number = account.account_number
    return number[-last:] if number else None


def _card_for(cards: list[SynthCard], account_pk: int) -> int | None:
    for c in cards:
        if c.account_pk == account_pk:
            return c.pk
    return None


def _ledger_leaf(account: SynthAccount) -> str:
    """A stable leaf name for a ledger account, derived from the account label."""
    return account.label.split("(")[0].strip().replace(" ", "")
