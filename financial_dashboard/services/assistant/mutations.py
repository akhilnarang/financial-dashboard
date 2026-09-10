"""Atomic transaction mutations used by the conversational assistant."""

import json
import re
from typing import NamedTuple, cast

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import (
    AuditAction,
    Account,
    CategoryReviewDecision,
    MerchantRule,
    Setting,
    Transaction,
    utc_now,
)
from financial_dashboard.services.assistant.contracts import ApplyTransactionChanges
from financial_dashboard.services.assistant.intent_policy import (
    derive_merchant_pattern,
    has_ambiguous_intent,
    has_global_no_change,
    has_non_mutating_intent,
    instruction_view,
    merchant_rule_is_explicit,
    mentions_category,
    negates_target,
    parse_instruction,
)
from financial_dashboard.services.categorization.manual import (
    CategoryDirectionPolicy,
    assign_category_no_commit,
    resolve_assistant_category_slug,
)
from financial_dashboard.services.categorization.decision_lifecycle import (
    supersede_active_decisions,
)
from financial_dashboard.services.categorization.merchant_rules import add_merchant_rule
from financial_dashboard.services.categorization.vocabulary import refresh_vocab_cache
from financial_dashboard.services.categorization.normalize import normalize_text
from financial_dashboard.services.categorization.polarity import resolve_direction
from financial_dashboard.services.transactions import (
    set_transaction_excluded_no_commit,
    update_transaction_note_no_commit,
)


DirectionPolicy = CategoryDirectionPolicy


class MutationResult(NamedTuple):
    transaction_id: int
    before: dict[str, object]
    after: dict[str, object]
    action_ids: tuple[int, ...]
    merchant_rule_pattern: str | None = None
    merchant_rule_category: str | None = None


class MutationRejected(ValueError):
    """The requested assistant mutation was not safe to apply."""


_NEGATION = (
    r"(?:do not|don t|never|no longer|stop|not|should not|shouldn t|"
    r"cannot|can t|will not|won t|would not|wouldn t|must not|mustn t|"
    r"could not|couldn t|may not|might not)"
)


def _category_assignment_is_negated(text: str) -> bool:
    """Detect a denied assignment without confusing it with denied creation."""
    for clause in re.split(r"[,.;:!?\n]", text):
        normalized_clause = normalize_text(clause)
        if re.search(
            rf"\b{_NEGATION}\b(?:\s+\w+){{0,7}}\s+"
            r"(?:categorize|categorise|classify)\b",
            normalized_clause,
        ) or re.search(
            rf"\b{_NEGATION}\b(?:\s+\w+){{0,5}}\s+"
            r"(?:set|change|assign|use)\b(?:\s+\w+){0,5}\s+category\b",
            normalized_clause,
        ):
            return True
    return False


def _cashflow_polarity_is_explicit(text: str, excluded: bool) -> bool:
    negative_exclude = bool(
        re.search(
            rf"\b{_NEGATION}\b"
            r"(?:\s+\w+){0,5}\s+exclude(?:d)?\b",
            text,
        )
    )
    negative_include = bool(
        re.search(
            rf"\b{_NEGATION}\b"
            r"(?:\s+\w+){0,5}\s+include(?:d)?\b",
            text,
        )
    )
    if excluded:
        if negative_exclude:
            return False
        return bool(
            re.search(r"\bexclude(?:d)?\b", text)
            or re.search(r"\bremove\b.{0,32}\bcash ?flow\b", text)
            or negative_include
            or re.search(r"\bcash ?flow exclusion\b.{0,16}\b(?:on|true|yes)\b", text)
        )
    if negative_include:
        return False
    return bool(
        re.search(r"\binclude(?:d)?\b", text)
        or re.search(r"\b(?:add|restore|count)\b.{0,32}\bcash ?flow\b", text)
        or negative_exclude
        or re.search(r"\bcash ?flow exclusion\b.{0,16}\b(?:off|false|no)\b", text)
    )


async def _category_evidence(
    session: AsyncSession, instruction: str, category: str
) -> str:
    """Resolve spelling only in an explicit category field, never arbitrary prose."""
    if mentions_category(instruction, category):
        return category
    for match in re.finditer(
        r"\bcategory\s*:?\s+(?:to\s+)?([\w-]+)", instruction, re.IGNORECASE
    ):
        spelling = match.group(1)
        resolved, _ = await resolve_assistant_category_slug(session, spelling)
        if resolved == category:
            return spelling
    return category


async def _validate_ordinary_intent(
    session: AsyncSession,
    request: ApplyTransactionChanges,
    current_user_message: str,
) -> None:
    """Bind ordinary patches to declarative current-turn evidence."""
    raw = current_user_message.strip()
    instruction = parse_instruction(raw)
    instruction_text = instruction.text
    normalized = normalize_text(instruction_text)
    changes = request.changes
    if instruction.note_requires_category and not (
        changes.note is not None
        and changes.note.op == "set"
        and changes.category is not None
        and changes.category.op == "set"
    ):
        raise MutationRejected(
            "note/category shorthand requires both note and category changes"
        )
    if instruction.note_shorthand and (
        "?" in raw
        or has_global_no_change(raw)
        or has_ambiguous_intent(raw)
        or has_non_mutating_intent(raw)
    ):
        raise MutationRejected(
            "questions or uncertain shorthand cannot change transaction data"
        )
    if has_global_no_change(instruction_text):
        raise MutationRejected("the current message forbids transaction changes")
    if has_ambiguous_intent(instruction_text):
        raise MutationRejected("uncertain instructions cannot change transaction data")
    if "?" in instruction_text or has_non_mutating_intent(instruction_text):
        raise MutationRejected("questions cannot change transaction data")
    if changes.note is not None:
        raw_note_value = changes.note.value if changes.note.op == "set" else ""
        note_value = (
            normalize_text(raw_note_value).strip() if changes.note.op == "set" else ""
        )
        intent_without_payload = instruction_text
        if negates_target(intent_without_payload, "note"):
            raise MutationRejected(
                "negated instructions cannot change transaction data"
            )
        if changes.note.op == "set":
            bounded_note = normalize_text(instruction.note_payload or "").strip()
            if not bounded_note or note_value != bounded_note:
                raise MutationRejected(
                    "note value must match the application-bounded note payload"
                )
        elif not re.search(r"\b(?:clear|remove|delete)\b.*\bnote\b", normalized):
            raise MutationRejected("clearing a note requires current-message intent")
    if changes.category is not None:
        if changes.category.op == "set":
            evidence = await _category_evidence(
                session, instruction_text, changes.category.value or ""
            )
            value = normalize_text(evidence).replace("_", " ")
            if negates_target(
                instruction_text, value
            ) or _category_assignment_is_negated(instruction_text):
                raise MutationRejected(
                    "negated instructions cannot change transaction data"
                )
            if not instruction.note_shorthand and not mentions_category(
                normalized, value
            ):
                raise MutationRejected(
                    "category value must be supported by the current message"
                )
        else:
            if negates_target(instruction_text, "category"):
                raise MutationRejected(
                    "negated instructions cannot change transaction data"
                )
            if not re.search(
                r"\b(?:clear|remove|delete|uncategorize|uncategorise)\b.*\bcategory\b",
                normalized,
            ):
                raise MutationRejected(
                    "clearing a category requires current-message intent"
                )
    if changes.exclude_from_cashflow is not None and not _cashflow_polarity_is_explicit(
        normalized, changes.exclude_from_cashflow.value
    ):
        raise MutationRejected("cashflow exclusion requires current-message intent")


def _snapshot(txn: Transaction) -> dict[str, object]:
    return {
        "note": txn.note,
        "category": txn.category,
        "exclude_from_cashflow": txn.exclude_from_cashflow,
        "category_method": txn.category_method,
        "category_model": txn.category_model,
        "review_status": txn.review_status,
    }


async def apply_transaction_changes(
    session: AsyncSession,
    request: ApplyTransactionChanges,
    *,
    current_user_message: str,
    interaction_id: int | None = None,
    direction_policy: DirectionPolicy = DirectionPolicy.INFERRED_STRICT,
    consumed_decision_id: int | None = None,
    confirmed_pending: bool = False,
) -> MutationResult:
    """Apply the complete patch inside a savepoint owned by the caller.

    A rejected later field must roll back earlier field changes while leaving
    the outer transaction usable for the audited error response.
    """
    # Python's sqlite3 legacy mode does not BEGIN for SELECT, so a first
    # SAVEPOINT can otherwise become the outermost transaction. A no-op write
    # against this non-projection setting opens the caller-owned transaction
    # without dirtying Paisa's transaction/account revision triggers.
    await session.execute(
        update(Setting)
        .where(Setting.key == "category_vocab_version")
        .values(value=Setting.value)
    )
    try:
        async with session.begin_nested():
            return await _apply_transaction_changes(
                session,
                request,
                current_user_message=current_user_message,
                interaction_id=interaction_id,
                direction_policy=direction_policy,
                consumed_decision_id=consumed_decision_id,
                confirmed_pending=confirmed_pending,
            )
    except MutationRejected:
        raise
    except ValueError as exc:
        raise MutationRejected(str(exc)) from exc


async def _apply_transaction_changes(
    session: AsyncSession,
    request: ApplyTransactionChanges,
    *,
    current_user_message: str,
    interaction_id: int | None = None,
    direction_policy: DirectionPolicy = DirectionPolicy.INFERRED_STRICT,
    consumed_decision_id: int | None = None,
    confirmed_pending: bool = False,
) -> MutationResult:
    """Validate then apply one complete assistant patch without committing."""
    txn = await session.get(Transaction, request.transaction_id)
    if txn is None:
        raise MutationRejected("transaction not found")
    changes = request.changes
    instruction_text = instruction_view(current_user_message)
    if (
        changes.note is None
        and changes.category is None
        and changes.exclude_from_cashflow is None
        and request.merchant_rule is None
    ):
        raise MutationRejected("empty transaction change set")
    if not confirmed_pending:
        await _validate_ordinary_intent(session, request, current_user_message)
    before = _snapshot(txn)
    merchant_before: dict[str, object] | None = None
    merchant_after: dict[str, object] | None = None

    # Resolve and validate the complete request before changing any ORM field.
    if (
        changes.note is not None
        and changes.note.op == "set"
        and not isinstance(changes.note.value, str)
    ):
        raise MutationRejected("note must be text")
    if changes.exclude_from_cashflow is not None and not isinstance(
        changes.exclude_from_cashflow.value, bool
    ):
        raise MutationRejected("exclude_from_cashflow must be boolean")

    # A merchant-rule-only request still targets the transaction's existing
    # category.  Keeping the local value in sync with the row lets the
    # category equality guard validate the rule without requiring a redundant
    # category patch in the same request.
    category = txn.category or ""
    if changes.category is not None:
        category = changes.category.value if changes.category.op == "set" else ""
        if changes.category.op == "set" and category is not None:
            resolved, corrected = await resolve_assistant_category_slug(
                session, category
            )
            if resolved is None:
                raise MutationRejected("invalid category")
            category = resolved
            if corrected or category != changes.category.value:
                changes = changes.model_copy(
                    update={
                        "category": changes.category.model_copy(
                            update={"value": category}
                        )
                    }
                )
                request = request.model_copy(update={"changes": changes})
            if direction_policy == DirectionPolicy.INFERRED_STRICT:
                account_type = None
                if txn.account_id is not None:
                    account = await session.get(Account, txn.account_id)
                    account_type = account.type if account is not None else None
                if (
                    resolve_direction(category, txn.direction, account_type).slug
                    != category
                ):
                    raise MutationRejected("invalid category for transaction direction")

    merchant_pattern: str | None = None
    merchant_category: str | None = None
    if request.merchant_rule is not None:
        merchant_category, _ = await resolve_assistant_category_slug(
            session, request.merchant_rule.category
        )
        if merchant_category is None or merchant_category != category:
            raise MutationRejected(
                "merchant rule category must be the transaction category"
            )
        if not confirmed_pending and not merchant_rule_is_explicit(
            instruction_text,
            request.merchant_rule.intent_evidence,
            await _category_evidence(
                session, instruction_text, request.merchant_rule.category
            ),
        ):
            raise MutationRejected("merchant rule requires current-message intent")
        merchant_pattern = derive_merchant_pattern(txn.counterparty)
        existing_rule = await session.scalar(
            select(MerchantRule).where(MerchantRule.pattern == merchant_pattern)
        )
        if existing_rule is not None:
            merchant_before = {
                "id": existing_rule.id,
                "pattern": existing_rule.pattern,
                "category": existing_rule.category,
                "active": existing_rule.active,
                "priority": existing_rule.priority,
            }
        if merchant_category != request.merchant_rule.category:
            request = request.model_copy(
                update={
                    "merchant_rule": request.merchant_rule.model_copy(
                        update={"category": merchant_category}
                    )
                }
            )

    if changes.note is not None:
        if changes.note.op == "clear":
            await update_transaction_note_no_commit(session, txn.id, "")
        else:
            await update_transaction_note_no_commit(session, txn.id, changes.note.value)

    if changes.category is not None:
        ok, _ = await assign_category_no_commit(
            session,
            txn.id,
            cast(str, category),
            actor="telegram_assistant",
            create=False,
            direction_policy=direction_policy,
            preserve_decision_id=consumed_decision_id,
        )
        if not ok:
            raise MutationRejected("invalid category")
        txn.category_model = "manual:telegram_assistant"

    if changes.exclude_from_cashflow is not None:
        value = changes.exclude_from_cashflow.value
        await set_transaction_excluded_no_commit(session, txn.id, value)

    if request.merchant_rule is not None:
        assert merchant_pattern is not None and merchant_category is not None
        await add_merchant_rule(
            session, merchant_pattern, merchant_category, priority=100
        )
        saved_rule = await session.scalar(
            select(MerchantRule).where(MerchantRule.pattern == merchant_pattern)
        )
        merchant_after = {
            "id": saved_rule.id if saved_rule is not None else 0,
            "pattern": merchant_pattern,
            "category": merchant_category,
            "active": True,
            "priority": 100,
        }

    await session.flush()
    # A successful category assignment makes all other candidate buttons stale.
    # The callback path passes its claimed decision so it is consumed together.
    if changes.category is not None:
        await supersede_active_decisions(
            session, txn.id, preserve_decision_id=consumed_decision_id
        )
        if consumed_decision_id is not None:
            consumed = await session.execute(
                update(CategoryReviewDecision)
                .where(
                    CategoryReviewDecision.id == consumed_decision_id,
                    CategoryReviewDecision.transaction_id == txn.id,
                    CategoryReviewDecision.status == "active",
                )
                .values(
                    status="consumed",
                    consumed_at=utc_now(),
                    selected_slug=txn.category,
                )
            )
            if cast(CursorResult, consumed).rowcount != 1:
                raise MutationRejected("category decision is no longer active")
    after = _snapshot(txn)
    action_ids: list[int] = []
    action_specs = [("transaction_change", before, after)]
    if request.merchant_rule is not None:
        action_specs.append(("merchant_rule", merchant_before, merchant_after))
    for action_type, old, new in action_specs:
        action = AuditAction(
            interaction_id=interaction_id,
            action_type=action_type,
            target_type="transaction"
            if action_type == "transaction_change"
            else "merchant_rule",
            target_id=txn.id
            if action_type == "transaction_change"
            else int(cast(int, merchant_after["id"]))
            if merchant_after and "id" in merchant_after
            else 0,
            arguments_json=json.dumps(
                request.model_dump(mode="json"), sort_keys=True, default=str
            ),
            before_json=json.dumps(old, sort_keys=True, default=str)
            if old is not None
            else None,
            after_json=json.dumps(new, sort_keys=True, default=str)
            if new is not None
            else None,
            undo_status="available" if action_type == "merchant_rule" else None,
        )
        session.add(action)
        await session.flush()
        action_ids.append(action.id)
    return MutationResult(
        txn.id,
        before,
        after,
        tuple(action_ids),
        cast(str, merchant_after["pattern"]) if merchant_after else None,
        cast(str, merchant_after["category"]) if merchant_after else None,
    )


async def undo_merchant_rule(
    session: AsyncSession, action_id: int, *, interaction_id: int | None = None
) -> bool:
    """CAS-claim and reverse one merchant-rule action in this transaction."""
    action = await session.get(AuditAction, action_id)
    if action is None or action.action_type != "merchant_rule":
        return False
    claimed = await session.execute(
        update(AuditAction)
        .where(AuditAction.id == action_id, AuditAction.undo_status == "available")
        .values(undo_status="claimed")
    )
    if cast(CursorResult, claimed).rowcount != 1:
        return False
    before = json.loads(action.before_json) if action.before_json else None
    after = json.loads(action.after_json) if action.after_json else None
    if not after or not isinstance(after, dict):
        action.undo_status = "stale"
        return False
    pattern = after.get("pattern")
    rule = await session.scalar(
        select(MerchantRule).where(MerchantRule.pattern == pattern)
    )
    if (
        rule is None
        or {
            "id": rule.id,
            "pattern": rule.pattern,
            "category": rule.category,
            "active": rule.active,
            "priority": rule.priority,
        }
        != after
    ):
        action.undo_status = "stale"
        return False
    if before is None:
        await session.delete(rule)
    else:
        rule.category = before["category"]
        rule.active = before["active"]
        rule.priority = before["priority"]
    action.undo_status = "undone"
    action.undone_at = utc_now()
    undo_action = AuditAction(
        interaction_id=interaction_id,
        action_type="merchant_rule_undo",
        target_type="merchant_rule",
        target_id=rule.id if rule is not None else int(after.get("id", 0)),
        arguments_json=json.dumps({"actor": "telegram_assistant"}),
        before_json=json.dumps(after, sort_keys=True),
        after_json=json.dumps(before, sort_keys=True) if before is not None else None,
    )
    session.add(undo_action)
    await session.flush()
    action.undone_by_interaction_id = interaction_id
    return True


async def refresh_caches_after_commit(session: AsyncSession) -> None:
    """Called only after commit by the interaction owner."""
    await refresh_vocab_cache(session)
    from financial_dashboard.services.categorization.merchant_rules import (
        load_merchant_rules,
    )

    await load_merchant_rules(session)
