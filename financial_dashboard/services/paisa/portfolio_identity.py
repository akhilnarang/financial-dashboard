"""Stable, non-reversible ledger identity for a CAS portfolio.

A CAS ``portfolio_key`` is a PAN — a government identifier. The valuation-only
fallback needs a *per-portfolio* account name so two portfolios stay distinct in
the generated journal, but the journal is a file on disk that may be synced to a
Paisa instance, committed, or shared. Someone holding only that file must not be
able to recover the PAN from the account name.

The identity is therefore a **keyed** HMAC of the normalized portfolio key under
a per-installation secret:

    token = base32(HMAC-SHA256(secret, normalized_portfolio_key))[:16]

Why keyed rather than a plain hash: the PAN space is small and structured (five
letters, four digits, one letter), so an *unkeyed* digest — truncated or not —
falls to an offline brute-force enumeration in seconds. The secret is never
written to the journal, so without it the token is not invertible. 16 base32
characters carry 80 bits, well past the collision floor for the handful of
portfolios one installation holds.

**Projection never writes.** The secret is seeded once by the migration in
:mod:`financial_dashboard.db.init_db` (and is created on demand by CAS ingestion
for an installation that predates it), so a read-only projection only ever reads
it. When no secret exists the projection does not invent one and does not fall
back to a bare hash: it degrades to the single shared, portfolio-less account
(:func:`shared_valuation_account`), which leaks nothing at the cost of merging
portfolios in the ledger.
"""

import base64
import hashlib
import hmac
import secrets

#: The settings key holding the per-installation HMAC secret (hex text).
PORTFOLIO_TOKEN_SECRET_KEY = "paisa.portfolio_token_secret"

#: Secret size. 32 bytes matches HMAC-SHA256's block-derived key strength.
_SECRET_BYTES = 32

#: Base32 characters kept from the digest. 16 chars = 80 bits, comfortably above
#: the requested 64-bit floor and collision-free for realistic portfolio counts.
_TOKEN_CHARS = 16

#: Prefix so the segment always starts with a letter (a bare base32 token may
#: start with a digit, which :func:`sanitize_commodity`-style rules and a human
#: reader both prefer to avoid) and reads as an opaque handle, not a value.
_TOKEN_PREFIX = "P-"


def new_portfolio_token_secret() -> str:
    """A fresh per-installation secret as hex text (for the settings row)."""
    return secrets.token_hex(_SECRET_BYTES)


def normalize_portfolio_key(portfolio_key: str) -> str:
    """The canonical form the token is derived from.

    Matches the normalization every other CAS reader applies
    (``portfolio_key.strip().upper()``) so the same portfolio always yields the
    same token regardless of the casing a given upload stored.
    """
    return (portfolio_key or "").strip().upper()


def portfolio_token(portfolio_key: str, secret: str | None) -> str | None:
    """The stable opaque token for *portfolio_key*, or ``None`` without a secret.

    Deterministic for a given (secret, portfolio_key): the same installation
    regenerates byte-identical journals across restarts. Returns ``None`` when
    the secret is missing or blank so the caller degrades to the shared account
    rather than emitting an unkeyed — and therefore reversible — digest.
    """
    normalized = normalize_portfolio_key(portfolio_key)
    if not normalized or not secret or not secret.strip():
        return None
    digest = hmac.new(
        secret.strip().encode("utf-8"), normalized.encode("utf-8"), hashlib.sha256
    ).digest()
    token = base64.b32encode(digest).decode("ascii").rstrip("=")[:_TOKEN_CHARS]
    return f"{_TOKEN_PREFIX}{token}"
