"""Orchestrates rule + LLM categorization for a single transaction, and the
selection query used by the sweep."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import Account, Transaction, utc_now
from financial_dashboard.services.categorization import gemini, openai_provider
from financial_dashboard.services.categorization.fewshot import get_similar_examples
from financial_dashboard.services.categorization.gemini import NEEDS_REVIEW
from financial_dashboard.services.categorization.hashing import (
    build_input_payload,
    compute_input_hash,
)
from financial_dashboard.services.categorization.rules import (
    RULESET_VERSION,
    load_rule_config,
    match_rules,
)
from financial_dashboard.services.categorization.polarity import resolve_direction
from financial_dashboard.services.categorization.self_transfer import (
    apply_reference_self_transfer_rule,
)
from financial_dashboard.services.categorization.vocabulary import (
    get_active_slugs,
    get_vocab_version,
)
from financial_dashboard.services.settings import (
    get_gemini_api_key,
    get_openai_api_key,
    get_openai_base_url,
    get_redact_name_tokens,
    get_setting,
)


async def resolve_account_type(session: AsyncSession, txn: Transaction) -> str | None:
    if txn.account_id is None:
        return None
    account = await session.get(Account, txn.account_id)
    return account.type if account else None


def _active_model_name() -> str:
    """Return the model identifier for whichever provider is currently selected."""
    provider = get_setting("categorization.llm_provider") or "gemini"
    if provider == "openai":
        return get_setting("openai.model") or "gpt-4o-mini"
    return get_setting("gemini.model") or gemini.MODEL_DEFAULT


async def _llm_classify(*, fields, examples, active_slugs) -> gemini.GeminiResult:
    # Thin indirection so tests can monkeypatch the network call.
    provider = get_setting("categorization.llm_provider") or "gemini"
    name_tokens = get_redact_name_tokens()
    if provider == "openai":
        return await openai_provider.classify(
            fields=fields,
            examples=examples,
            active_slugs=active_slugs,
            api_key=get_openai_api_key(),
            model=get_setting("openai.model") or "gpt-4o-mini",
            base_url=get_openai_base_url(),
            name_tokens=name_tokens,
        )
    return await gemini.classify(
        fields=fields,
        examples=examples,
        active_slugs=active_slugs,
        api_key=get_gemini_api_key(),
        model=get_setting("gemini.model") or gemini.MODEL_DEFAULT,
        name_tokens=name_tokens,
    )


def _confidence_threshold() -> float:
    try:
        return float(get_setting("categorization.confidence_threshold") or "0.6")
    except ValueError, TypeError:
        return 0.6


async def categorize_one(
    session: AsyncSession, txn: Transaction, *, use_llm: bool
) -> str:
    if await apply_reference_self_transfer_rule(session, txn):
        return "rule"

    account_type = await resolve_account_type(session, txn)
    payload = build_input_payload(txn, account_type)
    input_hash = compute_input_hash(payload)
    fields = {
        "counterparty": txn.counterparty,
        "raw_description": txn.raw_description,
        "channel": txn.channel,
        "email_type": txn.email_type,
        "direction": txn.direction,
        "amount": str(txn.amount),
        "currency": txn.currency or "INR",
        # None when the transaction has no linked account. The rules layer needs
        # it: what a credit can possibly mean depends on the account it lands on.
        "account_type": account_type,
    }

    rule_hit = match_rules(fields, load_rule_config())
    if rule_hit is not None:
        txn.category = rule_hit.slug
        txn.category_method = "rule"
        txn.category_confidence = rule_hit.confidence
        txn.category_model = RULESET_VERSION
        txn.category_input_hash = input_hash
        txn.category_vocab_version = get_vocab_version()
        txn.categorized_at = utc_now()
        txn.review_status = None
        return "rule"

    if not use_llm:
        # Rule pass found no match. Mark the row 'pending_llm' (not left NULL) so
        # the rule pass never re-evaluates it and the "never-touched" set strictly
        # shrinks — this is what lets the backfill terminate with full coverage.
        # The LLM pass selects 'pending_llm' rows explicitly.
        txn.category_method = "pending_llm"
        txn.category_input_hash = input_hash
        txn.category_vocab_version = get_vocab_version()
        return "skip"

    # Empty input → don't spend an LLM call. Mark as method='llm'/unknown (NOT
    # 'rule') so the stale-vocab requeue in select_needs_work_stmt reconsiders it
    # after a new slug is added; the guard above re-skips cheaply (no API call)
    # unless enrichment later populated text.
    if not (txn.counterparty or txn.raw_description):
        txn.category = "unknown"
        txn.category_method = "llm"
        txn.category_confidence = 0.0
        txn.category_model = "empty-input"
        txn.category_input_hash = input_hash
        txn.category_vocab_version = get_vocab_version()
        txn.categorized_at = utc_now()
        txn.review_status = None
        return "llm"

    active_slugs = [
        s for s in (await get_active_slugs(session)) if s != "self_transfer"
    ]
    examples = await get_similar_examples(
        session, counterparty=txn.counterparty, direction=txn.direction, limit=5
    )
    result = await _llm_classify(
        fields=fields,
        examples=examples,
        active_slugs=active_slugs,
    )

    txn.category_method = "llm"
    txn.category_model = _active_model_name()
    txn.category_confidence = result.confidence
    txn.category_input_hash = input_hash
    txn.category_vocab_version = get_vocab_version()
    txn.categorized_at = utc_now()

    if result.slug == NEEDS_REVIEW or result.confidence < _confidence_threshold():
        resolved, _ = resolve_direction("unknown", txn.direction, account_type)
        txn.category = resolved
        txn.review_status = "pending"
        txn.review_reason = result.reason
    else:
        resolved, changed = resolve_direction(result.slug, txn.direction, account_type)
        txn.category = resolved
        if changed:
            # The model gave a directionally-impossible slug — it was confused,
            # so drop confidence and queue for human review instead of storing
            # the coerced fallback silently.
            txn.review_reason = f"direction fallback: model said {result.slug}"
            txn.category_confidence = min(result.confidence, 0.4)
            txn.review_status = "pending"
        else:
            txn.review_status = None
    return "llm"


def select_needs_work_stmt(*, llm: bool, limit: int):
    """Rows needing (re)categorization.

    State machine for category_method:
      NULL          -> never evaluated (rule pass picks it up)
      'pending_llm' -> rule pass ran, no rule hit, awaiting the LLM pass
      'rule'/'llm'/'manual' -> categorized (rule/manual are authoritative)

    llm=False: rule pass — rows never evaluated (method IS NULL).
    llm=True: LLM pass — 'pending_llm' rows, brand-new NULL rows, OR a prior LLM
    'unknown' whose vocabulary is now stale (a new slug may now fit).
    """
    current_vocab = get_vocab_version()
    if not llm:
        # Rule pass: only rows never evaluated at all.
        return (
            select(Transaction)
            .where(Transaction.category_method.is_(None))
            .order_by(Transaction.id)
            .limit(limit)
        )
    return (
        select(Transaction)
        .where(
            Transaction.category_method.is_(None)
            | (Transaction.category_method == "pending_llm")
            | (
                (Transaction.category == "unknown")
                & (Transaction.category_method == "llm")
                & (
                    Transaction.category_vocab_version.is_(None)
                    | (Transaction.category_vocab_version < current_vocab)
                )
            )
        )
        # Stable order so a persistently-failing head row can't monopolise the
        # LIMIT and starve the rest of the backlog every poll cycle.
        .order_by(Transaction.id)
        .limit(limit)
    )
