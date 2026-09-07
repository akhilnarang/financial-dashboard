"""Provider-independent assistant turn orchestration.

This module deliberately knows nothing about Telegram update objects.  The
transport resolves the trusted reply target and passes it here.
"""

import json
import datetime
import re
from decimal import Decimal, InvalidOperation
from collections.abc import Awaitable, Callable, Sequence
from typing import NamedTuple, cast

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.models import (
    Account,
    AuditInteraction,
    CategoryReviewDecision,
    Setting,
    TelegramMessageContext,
    TelegramOutboundDelivery,
    TelegramConversation,
    Transaction,
    as_utc,
    utc_now,
)
from financial_dashboard.services.assistant.context import (
    MAX_CONTEXT_CHARS,
    bounded_history,
    context_size,
    serialize_tool_result,
)
from financial_dashboard.services.assistant.contracts import (
    Answer,
    ApplyTransactionChanges,
    AssistantResponse,
    CategoryProposal,
    Clarification,
    Error,
    ToolCalls,
    GetTransaction,
    ListCategories,
    ListTransactions,
    parse_response,
)
from financial_dashboard.services.assistant.contracts import (
    CategoryCreationConfirmation,
    MerchantRuleConfirmation,
)
from financial_dashboard.services.assistant.intent_policy import (
    POLICY_VERSION,
    derive_merchant_pattern,
    has_ambiguous_intent,
    has_global_no_change,
    has_non_mutating_intent,
    is_direct_affirmative,
    parse_instruction,
)
from financial_dashboard.services.assistant.mutations import (
    DirectionPolicy,
    MutationRejected,
    MutationResult,
    apply_transaction_changes,
)
from financial_dashboard.services.assistant.prompt import PromptContext
from financial_dashboard.services.assistant.provider import (
    AssistantProvider,
    ProviderFailure,
)
from financial_dashboard.services.assistant.transaction_reads import (
    AssistantTransaction,
    get_transaction,
    list_transactions,
)
from financial_dashboard.services.categorization.polarity import resolve_direction
from financial_dashboard.services.categorization.hashing import (
    build_input_payload,
    compute_transaction_confirmation_hash,
    compute_input_hash,
)
from financial_dashboard.services.categorization.vocabulary import get_active_slugs
from financial_dashboard.services.categorization.decision_lifecycle import (
    supersede_active_decisions,
)
from financial_dashboard.services.categorization.normalize import normalize_text


class OrchestrationResult(NamedTuple):
    response: AssistantResponse
    mutation: MutationResult | None = None
    decision_id: int | None = None
    model_input_json: str | None = None
    model_output_json: str | None = None
    model_explanation: str | None = None
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    output_mode: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
    transaction_ids: tuple[int, ...] = ()


class AuthorizationChanged(MutationRejected):
    """The persisted Telegram recipient changed while work was in flight."""


class ProcessingLeaseLost(Exception):
    """Another worker reclaimed this interaction while a turn was running."""


def _request_direction_policy(
    request: ApplyTransactionChanges,
    user_message: str,
    default: DirectionPolicy,
) -> DirectionPolicy:
    patch = request.changes.category
    if patch is None or patch.op != "set" or patch.value is None:
        return default
    category_words = normalize_text(patch.value.replace("_", " ")).strip()
    instruction = parse_instruction(user_message)
    if (
        has_ambiguous_intent(instruction.text)
        or has_global_no_change(instruction.text)
        or has_non_mutating_intent(instruction.text)
        or (
            instruction.note_shorthand
            and (
                has_ambiguous_intent(user_message)
                or has_global_no_change(user_message)
                or has_non_mutating_intent(user_message)
            )
        )
    ):
        return default
    normalized_message = normalize_text(instruction.text)
    if re.search(rf"(?<!\w){re.escape(category_words)}(?!\w)", normalized_message):
        return DirectionPolicy.EXPLICIT_MANUAL_OVERRIDE
    return default


async def _lock_authorized_chat(
    session: AsyncSession, expected_chat_id: int | None
) -> None:
    """Acquire a real write fence and verify the persisted Telegram recipient."""
    if expected_chat_id is None:
        await session.execute(
            update(Setting)
            .where(Setting.key == "category_vocab_version")
            .values(value=Setting.value)
        )
        return
    fenced = cast(
        CursorResult,
        await session.execute(
            update(Setting)
            .where(
                Setting.key == "telegram.chat_id",
                Setting.value == str(expected_chat_id),
            )
            .values(value=Setting.value)
        ),
    )
    if fenced.rowcount == 1:
        return
    row = await session.get(Setting, "telegram.chat_id")
    if row is None:
        from financial_dashboard.services.settings import get_telegram_chat_id

        if get_telegram_chat_id() == expected_chat_id:
            # Test/fresh-install compatibility.  The flush is also the required
            # outer SQLite write transaction before any nested mutation savepoint.
            session.add(Setting(key="telegram.chat_id", value=str(expected_chat_id)))
            await session.flush()
            return
    raise AuthorizationChanged("Telegram authorization changed")


def _confirmation_question(
    pending: CategoryCreationConfirmation | MerchantRuleConfirmation,
    target: AssistantTransaction,
) -> str:
    """Render confirmations from validated application state, never model prose."""
    if isinstance(pending, CategoryCreationConfirmation):
        return (
            f"Create category '{pending.slug}' and assign it to "
            f"transaction #{pending.transaction_id}? Reply yes to confirm."
        )
    try:
        pattern = derive_merchant_pattern(target.counterparty)
    except ValueError as exc:
        raise MutationRejected(str(exc)) from exc
    return (
        f"Create a merchant rule for '{pattern}' using category "
        f"'{pending.category}' and assign it to transaction "
        f"#{pending.transaction_id}? Reply yes to confirm."
    )


async def current_transaction_state_hash(
    session: AsyncSession, transaction_id: int
) -> str | None:
    txn = await session.get(Transaction, transaction_id)
    if txn is None:
        return None
    account_type = None
    if txn.account_id is not None:
        account = await session.get(Account, txn.account_id)
        account_type = account.type if account is not None else None
    return compute_input_hash(build_input_payload(txn, account_type))


async def current_confirmation_state_hash(
    session: AsyncSession, transaction_id: int
) -> str | None:
    txn = await session.get(Transaction, transaction_id)
    if txn is None:
        return None
    account_type = None
    if txn.account_id is not None:
        account = await session.get(Account, txn.account_id)
        account_type = account.type if account is not None else None
    return compute_transaction_confirmation_hash(txn, account_type)


async def _dispatch_read(session: AsyncSession, call: object) -> object:
    if isinstance(call, GetTransaction):
        return await get_transaction(session, call.transaction_id)
    if isinstance(call, ListCategories):
        return await get_active_slugs(session)
    if isinstance(call, ListTransactions):
        try:
            date_from = (
                datetime.date.fromisoformat(call.date_from) if call.date_from else None
            )
            date_to = (
                datetime.date.fromisoformat(call.date_to) if call.date_to else None
            )
            amount = Decimal(call.amount) if call.amount else None
            if amount is not None and not amount.is_finite():
                raise ValueError("amount must be finite")
        except (ValueError, InvalidOperation) as exc:
            raise MutationRejected("invalid transaction filter") from exc
        return await list_transactions(
            session,
            limit=call.limit,
            offset=0,
            account_id=call.account_id,
            date_from=date_from,
            date_to=date_to,
            direction=call.direction,
            category=call.category,
            search=call.search,
            excluded=call.excluded,
            transaction_ids=call.transaction_ids,
            amount=amount,
            bank=call.bank,
            source=call.source,
            review_status=call.review_status,
            reference=call.reference,
        )
    raise MutationRejected("unsupported tool")


async def _persist_pending(
    session: AsyncSession,
    conversation_id: int | None,
    response: Clarification,
    *,
    source_interaction_id: int | None,
    transaction_state_hash: str,
    direction_policy: DirectionPolicy,
) -> None:
    if conversation_id is None or response.pending_confirmation is None:
        return
    pending = response.pending_confirmation
    conversation = await session.get(TelegramConversation, conversation_id)
    if conversation is None:
        return
    payload = {
        "action": pending.model_dump(),
        "policy_version": POLICY_VERSION,
        "direction_policy": direction_policy.value,
    }
    from financial_dashboard.services.assistant.conversations import (
        set_pending_confirmation,
    )

    set_pending_confirmation(
        conversation,
        payload,
        kind=pending.kind,
        source_interaction_id=source_interaction_id,
        expires_at=min(
            as_utc(conversation.expires_at) if conversation.expires_at else utc_now(),
            utc_now() + datetime.timedelta(hours=24),
        ),
        state_hash=transaction_state_hash,
    )


async def run_turn(
    session: AsyncSession,
    provider: AssistantProvider,
    *,
    user_message: str,
    transaction_id: int | None = None,
    conversation_id: int | None = None,
    interaction_id: int | None = None,
    history: Sequence[dict[str, str]] = (),
    categories: Sequence[str] = (),
    direction_policy: DirectionPolicy = DirectionPolicy.INFERRED_STRICT,
    authorized_chat_id: int | None = None,
    renew_lease: Callable[[], Awaitable[bool]] | None = None,
) -> OrchestrationResult:
    """Run at most four model steps and leave all writes uncommitted."""
    target = (
        await get_transaction(session, transaction_id)
        if transaction_id is not None
        else None
    )
    if transaction_id is not None and target is None:
        return OrchestrationResult(
            Error(
                outcome="error",
                message="That transaction no longer exists.",
                code="target_not_found",
            )
        )
    tool_results: list[dict[str, object]] = []
    calls_used = 0
    mutation: MutationResult | None = None
    read_transaction_ids: set[int] = set()
    prompt_records: list[str] = []
    output_records: list[dict[str, object]] = []
    last_result = None

    def completed(
        response: AssistantResponse,
        mutation_result: MutationResult | None = mutation,
        decision_id: int | None = None,
    ) -> OrchestrationResult:
        return OrchestrationResult(
            response,
            mutation_result,
            decision_id,
            json.dumps(prompt_records, ensure_ascii=False),
            json.dumps(output_records, ensure_ascii=False, default=str),
            getattr(response, "explanation", None),
            last_result.provider if last_result is not None else None,
            last_result.model if last_result is not None else None,
            last_result.prompt_version if last_result is not None else None,
            last_result.output_mode if last_result is not None else None,
            last_result.input_tokens if last_result is not None else None,
            last_result.output_tokens if last_result is not None else None,
            sum(  # ty: ignore[no-matching-overload]
                record.get("latency_ms", 0)
                for record in output_records
                if isinstance(record.get("latency_ms"), int)
            ),
            tuple(sorted(read_transaction_ids)),
        )

    while calls_used < 4:
        if calls_used and renew_lease is not None and not await renew_lease():
            raise ProcessingLeaseLost
        prompt_context = PromptContext(
            user_message=user_message,
            history=bounded_history(history),
            transaction=serialize_tool_result(target) if target is not None else None,
            tool_results=tool_results,
            categories=list(categories) or await get_active_slugs(session),
        )
        if context_size(prompt_context._asdict()) > MAX_CONTEXT_CHARS:
            return completed(
                Error(
                    outcome="error",
                    message="The conversation context is too large; please start a new thread.",
                    code="context_limit",
                )
            )
        calls_used += 1
        try:
            result = await provider.complete(prompt_context)
            last_result = result
            prompt_records.append(result.prompt)
            output_records.append(
                {"payload": dict(result.raw_payload), "latency_ms": result.latency_ms}
            )
            response = parse_response(result.response.model_dump())
        except ProviderFailure, ValueError, TypeError:
            return completed(
                Error(
                    outcome="error",
                    message="I couldn't safely interpret that request. Please rephrase it.",
                    code="invalid_model_output",
                )
            )
        if isinstance(response, (Answer, Clarification, CategoryProposal)):
            if isinstance(response, Clarification):
                pending = response.pending_confirmation
                if pending is not None:
                    if transaction_id != pending.transaction_id:
                        return completed(
                            Error(
                                outcome="error",
                                message="I need one unambiguous transaction target for that confirmation.",
                                code="ambiguous_target",
                            )
                        )
                    state_hash = await current_confirmation_state_hash(
                        session, pending.transaction_id
                    )
                    if state_hash is None:
                        return completed(
                            Error(
                                outcome="error",
                                message="That transaction no longer exists.",
                                code="target_not_found",
                            )
                        )
                    try:
                        if target is None:
                            raise MutationRejected("confirmation target is not trusted")
                        response = response.model_copy(
                            update={"question": _confirmation_question(pending, target)}
                        )
                        await _lock_authorized_chat(session, authorized_chat_id)
                        await _persist_pending(
                            session,
                            conversation_id,
                            response,
                            source_interaction_id=interaction_id,
                            transaction_state_hash=state_hash,
                            direction_policy=_request_direction_policy(
                                ApplyTransactionChanges(
                                    name="apply_transaction_changes",
                                    transaction_id=pending.transaction_id,
                                    changes={
                                        "category": {
                                            "op": "set",
                                            "value": (
                                                pending.slug
                                                if isinstance(
                                                    pending,
                                                    CategoryCreationConfirmation,
                                                )
                                                else pending.category
                                            ),
                                        }
                                    },
                                ),
                                user_message,
                                direction_policy,
                            ),
                        )
                    except MutationRejected as exc:
                        return completed(
                            Error(
                                outcome="error",
                                message=str(exc),
                                code="confirmation_rejected",
                            )
                        )
            if isinstance(response, CategoryProposal):
                try:
                    await _lock_authorized_chat(session, authorized_chat_id)
                    decision_id = await _save_proposal(
                        session, response, target, interaction_id
                    )
                except MutationRejected as exc:
                    return completed(
                        Error(
                            outcome="error",
                            message=str(exc),
                            code="proposal_rejected",
                        )
                    )
                return completed(response, None, decision_id)
            return completed(response, mutation)
        if isinstance(response, ToolCalls):
            mutation_calls = [
                call
                for call in response.calls
                if isinstance(call, ApplyTransactionChanges)
            ]
            if len(mutation_calls) > 1 or any(
                transaction_id != call.transaction_id for call in mutation_calls
            ):
                return completed(
                    Error(
                        outcome="error",
                        message="I need one unambiguous transaction target for a change.",
                        code="ambiguous_target",
                    )
                )
            pending_tool_results: list[dict[str, object]] = []
            pending_read_ids: set[int] = set()
            try:
                # Validate and execute every read before applying the sole mutation.
                # A rejected later call can therefore never leave an earlier write.
                for call in response.calls:
                    if isinstance(call, ApplyTransactionChanges):
                        continue
                    tool_result = await _dispatch_read(session, call)
                    if hasattr(tool_result, "id"):
                        pending_read_ids.add(cast(int, tool_result.id))
                    elif isinstance(tool_result, Sequence):
                        pending_read_ids.update(
                            row.id for row in tool_result if hasattr(row, "id")
                        )
                    pending_tool_results.append(
                        {
                            "tool": call.name,
                            "result": serialize_tool_result(tool_result),
                        }
                    )
                if mutation_calls:
                    call = mutation_calls[0]
                    await _lock_authorized_chat(session, authorized_chat_id)
                    try:
                        mutation = await apply_transaction_changes(
                            session,
                            call,
                            current_user_message=user_message,
                            interaction_id=interaction_id,
                            direction_policy=_request_direction_policy(
                                call, user_message, direction_policy
                            ),
                        )
                    except MutationRejected:
                        raise
            except MutationRejected as exc:
                return completed(
                    Error(
                        outcome="error",
                        message=str(exc),
                        code="mutation_rejected",
                    )
                )
            read_transaction_ids.update(pending_read_ids)
            tool_results.extend(pending_tool_results)
            if mutation is not None:
                return completed(response, mutation)
            continue
        return completed(response, mutation)
    return completed(
        Error(
            outcome="error",
            message="I couldn't resolve that in one turn. Please narrow the request.",
            code="tool_round_limit",
        ),
        mutation,
    )


async def claim_pending_confirmation(
    session: AsyncSession,
    conversation_id: int,
    state_hash: str,
    *,
    replied_to_interaction_id: int,
) -> ApplyTransactionChanges | None:
    """CAS-claim an application-authored pending action for a direct reply."""
    conversation = await session.get(TelegramConversation, conversation_id)
    if (
        conversation is None
        or conversation.pending_confirmation_kind == "consuming"
        or conversation.pending_confirmation_state_hash != state_hash
        or conversation.pending_confirmation_source_interaction_id
        != replied_to_interaction_id
        or conversation.pending_confirmation_expires_at is None
        or as_utc(conversation.pending_confirmation_expires_at) <= utc_now()
    ):
        return None
    payload = conversation.pending_confirmation_json
    if not payload:
        return None
    try:
        wrapper = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise MutationRejected("stored confirmation is invalid") from exc
    if wrapper.get("policy_version") != POLICY_VERSION:
        return None
    pending = wrapper.get("action")
    if not isinstance(pending, dict):
        return None
    try:
        if pending.get("kind") == "create_category":
            typed = CategoryCreationConfirmation.model_validate(pending)
            request = ApplyTransactionChanges(
                name="apply_transaction_changes",
                transaction_id=typed.transaction_id,
                changes={"category": {"op": "set", "value": typed.slug}},
                create_category={"slug": typed.slug, "intent_evidence": typed.slug},
            )
        elif pending.get("kind") == "merchant_rule":
            typed = MerchantRuleConfirmation.model_validate(pending)
            request = ApplyTransactionChanges(
                name="apply_transaction_changes",
                transaction_id=typed.transaction_id,
                changes={"category": {"op": "set", "value": typed.category}},
                merchant_rule={
                    "category": typed.category,
                    "intent_evidence": typed.category,
                },
            )
        else:
            return None
    except (ValueError, TypeError) as exc:
        raise MutationRejected("stored confirmation is invalid") from exc
    claimed = await session.execute(
        update(TelegramConversation)
        .where(
            TelegramConversation.id == conversation_id,
            TelegramConversation.pending_confirmation_state_hash == state_hash,
            TelegramConversation.pending_confirmation_source_interaction_id
            == replied_to_interaction_id,
            TelegramConversation.pending_confirmation_kind != "consuming",
            TelegramConversation.pending_confirmation_expires_at > utc_now(),
        )
        .values(pending_confirmation_kind="consuming")
        .execution_options(synchronize_session=False)
    )
    return request if cast(CursorResult, claimed).rowcount == 1 else None


async def run_pending_confirmation(
    session: AsyncSession,
    *,
    conversation_id: int,
    state_hash: str,
    user_message: str,
    replied_to_interaction_id: int,
    interaction_id: int | None = None,
    authorized_chat_id: int | None = None,
) -> MutationResult:
    """Consume a pending confirmation and apply it atomically."""
    if not is_direct_affirmative(user_message):
        raise MutationRejected("confirmation must be a direct affirmative")
    # Acquire SQLite's writer lock before reading the confirmation fingerprint.
    # Otherwise a concurrent manual edit could land between the freshness read
    # and the confirmation CAS, allowing a stale action to proceed.
    await _lock_authorized_chat(session, authorized_chat_id)
    fresh_hash = None
    conversation = await session.get(TelegramConversation, conversation_id)
    if conversation is not None and conversation.pending_confirmation_json:
        try:
            wrapper = json.loads(conversation.pending_confirmation_json)
        except json.JSONDecodeError as exc:
            raise MutationRejected("stored confirmation is invalid") from exc
        action = wrapper.get("action", {})
        if isinstance(action, dict) and isinstance(action.get("transaction_id"), int):
            fresh_hash = await current_confirmation_state_hash(
                session, action["transaction_id"]
            )
        try:
            pending_direction_policy = DirectionPolicy(
                wrapper.get("direction_policy", DirectionPolicy.INFERRED_STRICT)
            )
        except ValueError as exc:
            raise MutationRejected("stored confirmation is invalid") from exc
    else:
        pending_direction_policy = DirectionPolicy.INFERRED_STRICT
    if fresh_hash != state_hash:
        raise MutationRejected("transaction changed since confirmation was requested")
    async with session.begin_nested():
        request = await claim_pending_confirmation(
            session,
            conversation_id,
            state_hash,
            replied_to_interaction_id=replied_to_interaction_id,
        )
        if request is None:
            raise MutationRejected("confirmation is expired or already used")
        result = await apply_transaction_changes(
            session,
            request,
            current_user_message=user_message,
            interaction_id=interaction_id,
            confirmed_pending=True,
            direction_policy=pending_direction_policy,
        )
        await session.execute(
            update(TelegramConversation)
            .where(TelegramConversation.id == conversation_id)
            .values(
                pending_confirmation_kind=None,
                pending_confirmation_json=None,
                pending_confirmation_state_hash=None,
                pending_confirmation_source_interaction_id=None,
                pending_confirmation_expires_at=None,
            )
        )
    return result


async def _save_proposal(
    session: AsyncSession,
    response: CategoryProposal,
    target: object,
    interaction_id: int | None,
) -> int:
    target = cast(AssistantTransaction | None, target)
    if target is None:
        raise MutationRejected("category proposal target is not trusted")
    if response.transaction_id != target.id:
        raise MutationRejected("category proposal target is not trusted")
    active = set(await get_active_slugs(session))
    if any(candidate.slug not in active for candidate in response.candidates):
        raise MutationRejected("category proposal contains an unknown category")
    account_type = None
    if target.account_id is not None:
        account = await session.get(Account, target.account_id)
        account_type = account.type if account is not None else None
    for candidate in response.candidates:
        if (
            resolve_direction(candidate.slug, target.direction, account_type).slug
            != candidate.slug
        ):
            raise MutationRejected(
                "category proposal contains an incompatible category"
            )
    await supersede_active_decisions(session, target.id)
    state_hash = await current_transaction_state_hash(session, target.id)
    if state_hash is None:
        raise MutationRejected("transaction not found")
    decision = CategoryReviewDecision(
        transaction_id=target.id,
        source_interaction_id=interaction_id,
        category_input_hash=state_hash,
        candidates_json=json.dumps(
            [candidate.model_dump() for candidate in response.candidates],
            sort_keys=True,
        ),
        proposed_slug=response.candidates[0].slug,
        confidence=response.candidates[0].confidence,
        gate_reason=response.explanation,
    )
    session.add(decision)
    await session.flush()
    return decision.id


def _legacy_transaction_id(text: str) -> int | None:
    """Recognize only known bot notification headers from before message mapping."""
    import re
    from financial_dashboard.services.telegram import is_sms_duplicate_prompt

    first_line = text.splitlines()[0] if text else ""
    if is_sms_duplicate_prompt(first_line):
        return None
    known = first_line.startswith("🔍 Needs a category:") or first_line.startswith(
        ("🔴 ", "🟢 ", "🚫 ", "⏳ ", "🔁 ", "🔄 ")
    )
    if not known:
        return None
    match = re.search(r"#(\d+)\s*$", first_line)
    return int(match.group(1)) if match else None


async def _history(session: AsyncSession, conversation_id: int) -> list[dict[str, str]]:
    from financial_dashboard.db.models import AuditInteraction

    rows = list(
        (
            await session.scalars(
                select(AuditInteraction)
                .where(
                    AuditInteraction.conversation_id == conversation_id,
                    AuditInteraction.assistant_text.is_not(None),
                )
                .order_by(AuditInteraction.id.desc())
                .limit(12)
            )
        ).all()
    )
    turns: list[dict[str, str]] = []
    for row in reversed(rows):
        if row.user_text:
            turns.append({"role": "user", "text": row.user_text})
        if row.assistant_text:
            turns.append({"role": "assistant", "text": row.assistant_text})
    return turns


def _provider_from_application_settings() -> AssistantProvider:
    from financial_dashboard.services.assistant.provider import provider_from_settings
    from financial_dashboard.services.categorization import gemini
    from financial_dashboard.services.settings import (
        get_gemini_api_key,
        get_openai_api_key,
        get_openai_base_url,
        get_setting,
    )

    provider_name = get_setting("categorization.llm_provider") or "gemini"
    if provider_name == "openai":
        return provider_from_settings(
            provider="openai",
            api_key=get_openai_api_key(),
            model=get_setting("openai.model") or "gpt-4o-mini",
            base_url=get_openai_base_url(),
            reasoning_effort=get_setting("openai.reasoning_effort") or "",
        )
    return provider_from_settings(
        provider="gemini",
        api_key=get_gemini_api_key(),
        model=get_setting("gemini.model") or gemini.MODEL_DEFAULT,
    )


def _response_text(result: OrchestrationResult, transaction_id: int | None) -> str:
    response = result.response
    if result.mutation is not None:
        before = result.mutation.before
        after = result.mutation.after
        changed = [
            key
            for key in ("note", "category", "exclude_from_cashflow")
            if before.get(key) != after.get(key)
        ]
        label = ", ".join(changed) or "transaction"
        text = f"Saved {label} for #{result.mutation.transaction_id}."
        if result.mutation.merchant_rule_pattern:
            text += (
                f" Merchant rule '{result.mutation.merchant_rule_pattern}' uses "
                f"category '{result.mutation.merchant_rule_category}'."
            )
        return text
    if isinstance(response, Answer):
        return response.text
    if isinstance(response, Clarification):
        return response.question
    if isinstance(response, CategoryProposal):
        return response.explanation
    if isinstance(response, Error):
        return response.message
    if transaction_id is not None:
        return f"Reviewed transaction #{transaction_id}."
    return "Done."


def _transaction_result_text(transaction: AssistantTransaction) -> str:
    date = str(transaction.transaction_date or "unknown date")
    currency = transaction.currency or "INR"
    lines = [
        f"#{transaction.id} · {date} · {transaction.direction} "
        f"{transaction.amount} {currency}"
    ]
    if transaction.counterparty:
        lines.append(transaction.counterparty)
    details = [
        value
        for value in (
            f"category: {transaction.category}" if transaction.category else None,
            f"note: {transaction.note}" if transaction.note else None,
        )
        if value is not None
    ]
    if details:
        lines.append(" · ".join(details))
    return "\n".join(lines)


async def _queue_result(
    session: AsyncSession,
    *,
    interaction_id: int,
    worker_token: str,
    result: OrchestrationResult,
    transaction_id: int | None,
    recipient_chat_id: int,
    outcome_override: str | None = None,
) -> list[int]:
    from financial_dashboard.services.assistant.audit import finalize_interaction
    from financial_dashboard.services.assistant.delivery import make_delivery
    from financial_dashboard.services.assistant.rendering import split_plain_text

    response_text = _response_text(result, transaction_id)
    outcome = result.response.outcome
    if result.mutation is not None:
        outcome = "mutation"
    elif result.transaction_ids and isinstance(result.response, Answer):
        # A read-only answer that returned transaction rows is a query result
        # in the audit taxonomy, even though the provider response itself is
        # the generic ``answer`` variant.
        outcome = "query_result"
    if outcome_override is not None and not isinstance(result.response, Error):
        outcome = outcome_override
    if not await finalize_interaction(
        session,
        interaction_id,
        worker_token,
        status="ready_to_send",
        outcome=outcome,
        assistant_text=response_text,
        model_input_json=result.model_input_json,
        model_output_json=result.model_output_json,
        model_explanation=result.model_explanation,
        provider=result.provider,
        model=result.model,
        prompt_version=result.prompt_version,
        output_mode=result.output_mode,
        error_code=result.response.code if isinstance(result.response, Error) else None,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
    ):
        raise RuntimeError("assistant processing lease was lost")

    mapped_transaction_id = transaction_id
    if mapped_transaction_id is None and len(result.transaction_ids) == 1:
        mapped_transaction_id = result.transaction_ids[0]
    chunks = split_plain_text(response_text, limit=4000)
    delivery_ids: list[int] = []
    for ordinal, chunk in enumerate(chunks):
        delivery = make_delivery(
            recipient_chat_id=recipient_chat_id,
            text=chunk,
            ordinal=ordinal,
            interaction_id=interaction_id,
            transaction_id=mapped_transaction_id,
            parse_mode=None,
        )
        session.add(delivery)
        await session.flush()
        delivery.text = f"{delivery.text}\n\nRef: {delivery.delivery_token}"
        if ordinal == 0 and isinstance(result.response, CategoryProposal):
            delivery.reply_markup_json = json.dumps(
                [
                    [
                        {
                            "text": candidate.slug,
                            "callback_data": (
                                f"cat:v1:{result.decision_id}:{delivery.id}:{index}"
                            ),
                        }
                    ]
                    for index, candidate in enumerate(result.response.candidates)
                ],
                separators=(",", ":"),
            )
        if ordinal == 0 and result.mutation is not None:
            from financial_dashboard.db.models import AuditAction

            merchant_action = await session.scalar(
                select(AuditAction)
                .where(
                    AuditAction.id.in_(result.mutation.action_ids),
                    AuditAction.action_type == "merchant_rule",
                    AuditAction.undo_status == "available",
                )
                .limit(1)
            )
            if merchant_action is not None:
                delivery.reply_markup_json = json.dumps(
                    [
                        [
                            {
                                "text": "Undo merchant rule",
                                "callback_data": f"undo:v1:{merchant_action.id}",
                            }
                        ]
                    ],
                    separators=(",", ":"),
                )
        delivery_ids.append(delivery.id)
    if transaction_id is None and len(result.transaction_ids) > 1:
        for transaction_result_id in result.transaction_ids:
            transaction_result = await get_transaction(session, transaction_result_id)
            if transaction_result is None:
                continue
            for chunk in split_plain_text(
                _transaction_result_text(transaction_result), limit=4000
            ):
                delivery = make_delivery(
                    recipient_chat_id=recipient_chat_id,
                    text=chunk,
                    ordinal=len(delivery_ids),
                    interaction_id=interaction_id,
                    transaction_id=transaction_result.id,
                    parse_mode=None,
                )
                session.add(delivery)
                await session.flush()
                delivery.text = f"{delivery.text}\n\nRef: {delivery.delivery_token}"
                delivery_ids.append(delivery.id)
    return delivery_ids


async def _resolve_text_context(
    session: AsyncSession, message
) -> TelegramMessageContext | None:
    from financial_dashboard.services.assistant.message_context import (
        record_physical_message,
        recover_ref_context,
        resolve_reply,
    )

    reply = message.reply_to_message
    mapped = await resolve_reply(
        session, chat_id=message.chat_id, message_id=reply.message_id
    )
    if mapped is None and reply.text:
        mapped = await recover_ref_context(
            session,
            chat_id=message.chat_id,
            message_id=reply.message_id,
            message_text=reply.text,
        )
    if mapped is not None:
        return mapped
    transaction_id = _legacy_transaction_id(reply.text or reply.caption or "")
    if transaction_id is None or await session.get(Transaction, transaction_id) is None:
        return None
    return await record_physical_message(
        session,
        chat_id=message.chat_id,
        message_id=reply.message_id,
        context_kind="transaction_notification",
        transaction_id=transaction_id,
    )


async def _conversation_for_message(
    session: AsyncSession,
    *,
    chat_id: int,
    trigger: str,
    mapped,
):
    from financial_dashboard.services.assistant.conversations import (
        start_conversation,
        touch_conversation,
    )

    if trigger == "ask":
        return await start_conversation(session, chat_id=chat_id, started_by="ask")
    if mapped.conversation_id is not None:
        conversation = await session.get(TelegramConversation, mapped.conversation_id)
        if (
            conversation is not None
            and conversation.status == "active"
            and conversation.expires_at is not None
            and as_utc(conversation.expires_at) > utc_now()
        ):
            await touch_conversation(session, conversation)
            return conversation
        if conversation is not None:
            conversation.status = "expired"
            await session.flush()
            return conversation
    return await start_conversation(
        session,
        chat_id=chat_id,
        started_by="reply",
        transaction_id=mapped.transaction_id,
    )


async def _dispatch_interaction_outputs(interaction_id: int) -> None:
    from financial_dashboard.db import async_session
    from financial_dashboard.db.models import TelegramOutboundDelivery
    from financial_dashboard.services.telegram import dispatch_saved_delivery

    async with async_session() as session:
        ids = list(
            (
                await session.scalars(
                    select(TelegramOutboundDelivery.id)
                    .where(TelegramOutboundDelivery.interaction_id == interaction_id)
                    .order_by(TelegramOutboundDelivery.ordinal)
                )
            ).all()
        )
    for delivery_id in ids:
        await dispatch_saved_delivery(delivery_id)


async def _process_text_interaction(
    *,
    interaction_id: int,
    worker_token: str,
    conversation_id: int,
    transaction_id: int | None,
    user_text: str,
    replied_to_interaction_id: int | None,
    recipient_chat_id: int,
) -> None:
    """Finish one claimed text turn from persisted, restart-safe inputs."""
    from financial_dashboard.db import async_session

    async def renew_owned_lease() -> bool:
        from financial_dashboard.services.assistant.audit import (
            renew_processing_lease,
        )

        async with async_session() as lease_session:
            renewed = await renew_processing_lease(
                lease_session, interaction_id, worker_token
            )
            await lease_session.commit()
            return renewed

    try:
        async with async_session() as session:
            conversation = await session.get(TelegramConversation, conversation_id)
            if conversation is None:
                raise RuntimeError("assistant conversation no longer exists")
            result = (
                OrchestrationResult(
                    Error(
                        outcome="error",
                        message=(
                            "That conversation expired. Use /ask for a new search, "
                            "or reply to a transaction notification."
                        ),
                        code="conversation_expired",
                    )
                )
                if conversation.status != "active"
                else None
            )
            if result is None and (
                conversation.pending_confirmation_json
                and replied_to_interaction_id is not None
                and replied_to_interaction_id
                == conversation.pending_confirmation_source_interaction_id
                and is_direct_affirmative(user_text)
            ):
                try:
                    mutation = await run_pending_confirmation(
                        session,
                        conversation_id=conversation.id,
                        state_hash=conversation.pending_confirmation_state_hash or "",
                        user_message=user_text,
                        replied_to_interaction_id=replied_to_interaction_id,
                        interaction_id=interaction_id,
                        authorized_chat_id=recipient_chat_id,
                    )
                    result = OrchestrationResult(
                        Answer(outcome="answer", text="Saved."), mutation=mutation
                    )
                except AuthorizationChanged:
                    raise
                except MutationRejected as exc:
                    result = OrchestrationResult(
                        Error(
                            outcome="error",
                            message=str(exc),
                            code="confirmation_rejected",
                        )
                    )
            elif result is None and conversation.pending_confirmation_json:
                from financial_dashboard.services.assistant.conversations import (
                    clear_pending_confirmation,
                )

                clear_pending_confirmation(conversation)
                # Persist the deliberate dismissal before provider/tool rounds.
                # Their lease heartbeat uses an independent session so it is
                # visible to recovery; retaining this write in the turn session
                # would make that heartbeat contend with our own SQLite lock.
                await session.commit()
            if result is None:
                try:
                    provider = _provider_from_application_settings()
                except ProviderFailure, ValueError, TypeError:
                    result = OrchestrationResult(
                        Error(
                            outcome="error",
                            message="The assistant provider is not configured correctly.",
                            code="provider_configuration",
                        )
                    )
                else:
                    result = await run_turn(
                        session,
                        provider,
                        user_message=user_text,
                        transaction_id=transaction_id,
                        conversation_id=conversation.id,
                        interaction_id=interaction_id,
                        history=await _history(session, conversation.id),
                        authorized_chat_id=recipient_chat_id,
                        renew_lease=renew_owned_lease,
                    )
            await _lock_authorized_chat(session, recipient_chat_id)
            delivery_ids = await _queue_result(
                session,
                interaction_id=interaction_id,
                worker_token=worker_token,
                result=result,
                transaction_id=transaction_id,
                recipient_chat_id=recipient_chat_id,
            )
            await session.commit()
    except AuthorizationChanged:
        async with async_session() as session:
            from financial_dashboard.services.assistant.audit import (
                mark_authorization_changed,
            )

            await mark_authorization_changed(session, interaction_id)
            await session.commit()
        return
    except ProcessingLeaseLost:
        return
    if result.mutation is not None:
        try:
            async with async_session() as session:
                from financial_dashboard.services.assistant.mutations import (
                    refresh_caches_after_commit,
                )

                await refresh_caches_after_commit(session)
        except Exception:
            import logging

            logging.getLogger(__name__).exception(
                "Assistant mutation committed but cache refresh failed"
            )
    for delivery_id in delivery_ids:
        from financial_dashboard.services.telegram import dispatch_saved_delivery

        await dispatch_saved_delivery(delivery_id)


async def resume_claimed_interactions(*, bot=None, limit: int = 50) -> int:
    """Resume persisted text and attachment turns after an interrupted worker."""
    from financial_dashboard.db import AuditInteraction, async_session
    from financial_dashboard.services.assistant.audit import (
        claim_processing,
        mark_authorization_changed,
    )
    from financial_dashboard.services.assistant.message_context import resolve_reply
    from financial_dashboard.services.settings import get_telegram_chat_id

    triggers = ["ask", "reply", "category_button", "undo"]
    if bot is not None:
        triggers.append("attachment")
    async with async_session() as session:
        ids = list(
            (
                await session.scalars(
                    select(AuditInteraction.id)
                    .where(
                        AuditInteraction.status == "claimed",
                        AuditInteraction.trigger.in_(triggers),
                    )
                    .order_by(AuditInteraction.id)
                    .limit(limit)
                )
            ).all()
        )
    resumed = 0
    for interaction_id in ids:
        async with async_session() as session:
            interaction, worker_token = await claim_processing(session, interaction_id)
            if interaction is None:
                await session.rollback()
                continue
            if interaction.inbound_chat_id != get_telegram_chat_id():
                await mark_authorization_changed(session, interaction.id)
                await session.commit()
                continue
            trigger = interaction.trigger
            conversation_id = interaction.conversation_id
            if conversation_id is None and trigger in {"ask", "reply", "attachment"}:
                interaction.status = "failed"
                interaction.outcome = "error"
                interaction.error_code = "conversation_missing"
                interaction.worker_token = None
                interaction.processing_lease_until = None
                await session.commit()
                continue
            replied_to_interaction_id = None
            if interaction.reply_to_message_id is not None:
                mapped = await resolve_reply(
                    session,
                    chat_id=interaction.inbound_chat_id,
                    message_id=interaction.reply_to_message_id,
                )
                replied_to_interaction_id = (
                    mapped.interaction_id if mapped is not None else None
                )
            user_text = interaction.user_text or ""
            transaction_id = interaction.transaction_id
            recipient_chat_id = interaction.inbound_chat_id
            caption = interaction.user_text
            attachment_payload = None
            if trigger == "attachment":
                try:
                    attachment_payload = json.loads(
                        interaction.inbound_payload_json or ""
                    )
                    if not isinstance(attachment_payload.get("file_id"), str):
                        raise ValueError
                except ValueError, json.JSONDecodeError, AttributeError:
                    interaction.status = "failed"
                    interaction.outcome = "error"
                    interaction.error_code = "invalid_attachment_payload"
                    interaction.worker_token = None
                    interaction.processing_lease_until = None
                    await session.commit()
                    continue
            await session.commit()
        if trigger == "attachment":
            assert bot is not None and attachment_payload is not None
            await _process_attachment_interaction(
                bot=bot,
                interaction_id=interaction_id,
                worker_token=worker_token,
                transaction_id=transaction_id,
                recipient_chat_id=recipient_chat_id,
                file_id=attachment_payload["file_id"],
                declared_size=attachment_payload.get("declared_size"),
                caption=caption,
            )
        elif trigger in {"category_button", "undo"}:
            await _process_callback_interaction(
                interaction_id=interaction_id,
                worker_token=worker_token,
                trigger=trigger,
                callback_data=user_text,
                recipient_chat_id=recipient_chat_id,
                physical_message_id=interaction.inbound_message_id,
            )
        else:
            assert conversation_id is not None
            await _process_text_interaction(
                interaction_id=interaction_id,
                worker_token=worker_token,
                conversation_id=conversation_id,
                transaction_id=transaction_id,
                user_text=user_text,
                replied_to_interaction_id=replied_to_interaction_id,
                recipient_chat_id=recipient_chat_id,
            )
        resumed += 1
    return resumed


async def handle_telegram_update(update, context, *, trigger: str) -> None:
    """Narrow Telegram transport entrypoint with durable claim and replay."""
    from financial_dashboard.db import async_session
    from financial_dashboard.services.assistant.audit import (
        claim_interaction,
        claim_processing,
        mark_authorization_changed,
    )
    from financial_dashboard.services.settings import get_telegram_chat_id

    query = update.callback_query
    message = update.message or (query.message if query is not None else None)
    if message is None or message.chat.id != get_telegram_chat_id():
        return
    if trigger in {"category_button", "undo"}:
        await _handle_assistant_callback(update, context=context, trigger=trigger)
        return

    mapped = None
    async with async_session() as session:
        if trigger != "ask":
            mapped = await _resolve_text_context(session, message)
            if mapped is None:
                return
        mapped_conversation_exists = bool(
            mapped is not None
            and mapped.conversation_id is not None
            and await session.get(TelegramConversation, mapped.conversation_id)
            is not None
        )
        conversation = await _conversation_for_message(
            session,
            chat_id=message.chat_id,
            trigger=trigger,
            mapped=mapped,
        )
        transaction_id = (
            mapped.transaction_id
            if mapped is not None and mapped.transaction_id is not None
            else conversation.transaction_id
        )
        if trigger == "ask" and context.args:
            user_text = " ".join(context.args).strip()
        elif trigger == "attachment":
            # A receipt caption is transaction data; preserve it verbatim.
            user_text = message.caption
        else:
            user_text = (message.text or "").strip()
        inbound_payload = None
        if trigger == "attachment":
            media = message.document or (message.photo[-1] if message.photo else None)
            if media is None:
                raise MutationRejected("attachment has no media")
            inbound_payload = json.dumps(
                {
                    "file_id": media.file_id,
                    "file_unique_id": media.file_unique_id,
                    "declared_size": media.file_size,
                    "mime_type": getattr(media, "mime_type", "image/jpeg"),
                },
                sort_keys=True,
            )
        interaction, is_new = await claim_interaction(
            session,
            telegram_update_id=str(update.update_id),
            chat_id=message.chat_id,
            message_id=message.message_id,
            reply_to_message_id=(
                message.reply_to_message.message_id
                if message.reply_to_message is not None
                else None
            ),
            trigger=trigger,
            user_text=user_text,
            inbound_payload_json=inbound_payload,
        )
        if is_new:
            interaction.conversation_id = conversation.id
            interaction.transaction_id = transaction_id
        else:
            conversation_was_created = (
                trigger == "ask" or not mapped_conversation_exists
            )
            if (
                conversation_was_created
                and conversation.id != interaction.conversation_id
            ):
                await session.delete(conversation)
            if interaction.conversation_id is not None:
                stored_conversation = await session.get(
                    TelegramConversation, interaction.conversation_id
                )
                if stored_conversation is not None:
                    conversation = stored_conversation
            transaction_id = interaction.transaction_id
        await session.commit()
        if not is_new and interaction.status in {
            "ready_to_send",
            "delivery_partial",
            "delivered",
            "delivery_failed",
        }:
            interaction_id = interaction.id
            if interaction.status in {"ready_to_send", "delivery_partial"}:
                await _dispatch_interaction_outputs(interaction_id)
            return
        interaction, worker_token = await claim_processing(session, interaction.id)
        if interaction is None:
            await session.rollback()
            return
        if interaction.inbound_chat_id != get_telegram_chat_id():
            await mark_authorization_changed(session, interaction.id)
            await session.commit()
            return
        await session.commit()

    if trigger == "attachment":
        await _process_attachment_interaction(
            bot=context.bot,
            interaction_id=interaction.id,
            worker_token=worker_token,
            transaction_id=transaction_id,
            recipient_chat_id=message.chat.id,
            file_id=media.file_id,
            declared_size=media.file_size,
            caption=message.caption,
        )
        return

    await _process_text_interaction(
        interaction_id=interaction.id,
        worker_token=worker_token,
        conversation_id=conversation.id,
        transaction_id=transaction_id,
        user_text=user_text,
        replied_to_interaction_id=(
            mapped.interaction_id if mapped is not None else None
        ),
        recipient_chat_id=message.chat.id,
    )


async def _process_attachment_interaction(
    *,
    bot,
    interaction_id: int,
    worker_token: str,
    transaction_id: int | None,
    recipient_chat_id: int,
    file_id: str,
    declared_size: int | None,
    caption: str | None,
) -> None:
    import asyncio
    import httpx
    from telegram.error import TelegramError
    from sqlalchemy.exc import SQLAlchemyError
    from financial_dashboard.db import async_session
    from financial_dashboard.services.transaction_attachments import (
        AttachmentError,
        attach_downloaded_attachment,
        cleanup_replaced_attachment,
        download_attachment,
        record_attachment_cleanup_warning,
        remove_attachment,
    )
    from financial_dashboard.services.telegram import dispatch_saved_delivery

    mutation = None
    stored = None
    committed = False
    # Apply the same 24-hour conversation gate used by text turns before any
    # Telegram download. An expired reply must not attach a file or replace a
    # transaction caption, even if the worker was claimed before expiry.
    async with async_session() as session:
        interaction = await session.get(AuditInteraction, interaction_id)
        conversation = (
            await session.get(TelegramConversation, interaction.conversation_id)
            if interaction is not None and interaction.conversation_id is not None
            else None
        )
        expired = (
            conversation is None
            or conversation.status != "active"
            or conversation.expires_at is None
            or as_utc(conversation.expires_at) <= utc_now()
        )
    if expired:
        result = OrchestrationResult(
            Error(
                outcome="error",
                message=(
                    "That conversation expired. Use /ask for a new search, "
                    "or reply to a transaction notification."
                ),
                code="conversation_expired",
            )
        )
    elif transaction_id is None:
        result = OrchestrationResult(
            Error(
                outcome="error",
                message="Reply to one transaction to attach a receipt.",
                code="target_required",
            )
        )
    else:
        try:
            telegram_file = await bot.get_file(file_id)
            if not telegram_file.file_path:
                raise AttachmentError("Telegram did not provide a receipt download URL")
            stored = await download_attachment(
                str(telegram_file.file_path),
                transaction_id=transaction_id,
                declared_size=declared_size,
            )
            async with async_session() as session:
                await _lock_authorized_chat(session, recipient_chat_id)
                mutation = await attach_downloaded_attachment(
                    session,
                    transaction_id,
                    stored,
                    caption=caption,
                    interaction_id=interaction_id,
                    worker_token=worker_token,
                )
                result = OrchestrationResult(
                    Answer(
                        outcome="answer", text=f"Attached receipt to #{transaction_id}."
                    )
                )
                delivery_ids = await _queue_result(
                    session,
                    interaction_id=interaction_id,
                    worker_token=worker_token,
                    result=result,
                    transaction_id=transaction_id,
                    recipient_chat_id=recipient_chat_id,
                    outcome_override="attachment",
                )
                await session.commit()
                committed = True
            try:
                if not cleanup_replaced_attachment(mutation.old_path):
                    async with async_session() as session:
                        await record_attachment_cleanup_warning(
                            session, mutation.audit_action_id, mutation.old_path or ""
                        )
            except Exception:
                import logging

                logging.getLogger(__name__).exception(
                    "Receipt attached but old-file cleanup reporting failed"
                )
            for delivery_id in delivery_ids:
                await dispatch_saved_delivery(delivery_id)
            return
        except asyncio.CancelledError:
            if stored is not None and not committed:
                try:
                    remove_attachment(stored.relative_path)
                except OSError:
                    import logging

                    logging.getLogger(__name__).exception(
                        "Failed to clean up cancelled receipt download"
                    )
            raise
        except AuthorizationChanged:
            if stored is not None:
                remove_attachment(stored.relative_path)
            async with async_session() as session:
                from financial_dashboard.services.assistant.audit import (
                    mark_authorization_changed,
                )

                await mark_authorization_changed(session, interaction_id)
                await session.commit()
            return
        except AttachmentError as exc:
            if stored is not None:
                remove_attachment(stored.relative_path)
            result = OrchestrationResult(
                Error(outcome="error", message=str(exc), code="attachment_failed")
            )
        except (
            OSError,
            httpx.HTTPError,
            TelegramError,
            RuntimeError,
            SQLAlchemyError,
        ):
            if committed:
                import logging

                logging.getLogger(__name__).exception(
                    "Receipt committed but post-commit delivery failed"
                )
                return
            if stored is not None:
                remove_attachment(stored.relative_path)
            result = OrchestrationResult(
                Error(
                    outcome="error",
                    message="I couldn't download or store that receipt.",
                    code="attachment_failed",
                )
            )
    async with async_session() as session:
        try:
            await _lock_authorized_chat(session, recipient_chat_id)
            delivery_ids = await _queue_result(
                session,
                interaction_id=interaction_id,
                worker_token=worker_token,
                result=result,
                transaction_id=transaction_id,
                recipient_chat_id=recipient_chat_id,
                outcome_override="attachment",
            )
            await session.commit()
        except AuthorizationChanged:
            await session.rollback()
            from financial_dashboard.services.assistant.audit import (
                mark_authorization_changed,
            )

            await mark_authorization_changed(session, interaction_id)
            await session.commit()
            return
    for delivery_id in delivery_ids:
        await dispatch_saved_delivery(delivery_id)


async def _process_callback_interaction(
    *,
    interaction_id: int,
    worker_token: str,
    trigger: str,
    callback_data: str,
    recipient_chat_id: int,
    physical_message_id: int | None,
) -> None:
    from financial_dashboard.db import async_session
    from financial_dashboard.services.assistant.delivery import (
        settle_delivery_from_callback,
        validate_delivery_proof,
    )
    from financial_dashboard.services.assistant.message_context import (
        record_physical_message,
        resolve_reply,
    )
    from financial_dashboard.services.assistant.mutations import undo_merchant_rule
    from financial_dashboard.services.categorization.review_decisions import (
        consume_decision,
    )
    from financial_dashboard.services.telegram import dispatch_saved_delivery

    async with async_session() as session:
        interaction = await session.get(AuditInteraction, interaction_id)
        if interaction is None:
            raise RuntimeError("assistant callback interaction no longer exists")
        try:
            await _lock_authorized_chat(session, recipient_chat_id)
            if trigger == "category_button":
                try:
                    _, _, raw_decision, raw_delivery, raw_index = callback_data.split(
                        ":"
                    )
                    decision_id = int(raw_decision)
                    delivery_id = int(raw_delivery)
                    index = int(raw_index)
                    if index < 0:
                        raise ValueError("negative category index")
                    decision = await session.get(CategoryReviewDecision, decision_id)
                    if decision is None:
                        raise ValueError("unknown category decision")
                    candidates = (
                        json.loads(decision.candidates_json) if decision else []
                    )
                    selected_slug = candidates[index].get(
                        "category", candidates[index].get("slug")
                    )
                except ValueError, IndexError, AttributeError, TypeError:
                    action = None
                else:
                    # Telegram callback delivery is proof that this exact
                    # choice message was rendered.  Establish that proof
                    # before applying the category; a stale or unrelated
                    # delivery must never authorize a mutation.
                    owner_kwargs = {
                        "category_review_decision_id": decision_id
                        if decision.source_interaction_id is None
                        else None,
                        "interaction_id": decision.source_interaction_id,
                    }
                    delivery_proven = await validate_delivery_proof(
                        session,
                        delivery_id,
                        recipient_chat_id=recipient_chat_id,
                        transaction_id=decision.transaction_id,
                        **owner_kwargs,
                    ) and await settle_delivery_from_callback(session, delivery_id)
                    action = (
                        await consume_decision(
                            session,
                            decision_id,
                            selected_slug=selected_slug,
                            category_input_hash=decision.category_input_hash,
                            interaction_id=interaction.id,
                            delivery_id=delivery_id,
                        )
                        if delivery_proven
                        else None
                    )
                if action is None:
                    result = OrchestrationResult(
                        Error(
                            outcome="error",
                            message="That category choice is stale.",
                            code="stale_choice",
                        )
                    )
                else:
                    interaction.transaction_id = action.target_id
                    if physical_message_id is not None and (
                        await resolve_reply(
                            session,
                            chat_id=recipient_chat_id,
                            message_id=physical_message_id,
                        )
                        is None
                    ):
                        source_delivery = await session.get(
                            TelegramOutboundDelivery, delivery_id
                        )
                        source_conversation_id = None
                        source_interaction_id = None
                        if (
                            source_delivery is not None
                            and source_delivery.interaction_id is not None
                        ):
                            source_interaction = await session.get(
                                AuditInteraction, source_delivery.interaction_id
                            )
                            if source_interaction is not None:
                                source_conversation_id = (
                                    source_interaction.conversation_id
                                )
                                source_interaction_id = source_interaction.id
                        await record_physical_message(
                            session,
                            chat_id=recipient_chat_id,
                            message_id=physical_message_id,
                            context_kind="category_review",
                            conversation_id=source_conversation_id,
                            transaction_id=action.target_id,
                            interaction_id=source_interaction_id,
                            outbound_delivery_id=delivery_id,
                        )
                    result = OrchestrationResult(
                        Answer(
                            outcome="answer",
                            text=f"Saved category {selected_slug} for #{action.target_id}.",
                        )
                    )
            else:
                try:
                    _, _, raw_action = callback_data.split(":")
                    undone = await undo_merchant_rule(
                        session, int(raw_action), interaction_id=interaction.id
                    )
                except ValueError:
                    undone = False
                result = OrchestrationResult(
                    Answer(outcome="answer", text="Merchant rule restored.")
                    if undone
                    else Error(
                        outcome="error",
                        message="That undo is stale or already used.",
                        code="stale_undo",
                    )
                )
            delivery_ids = await _queue_result(
                session,
                interaction_id=interaction.id,
                worker_token=worker_token,
                result=result,
                transaction_id=interaction.transaction_id,
                recipient_chat_id=recipient_chat_id,
                outcome_override=(
                    "category_choice" if trigger == "category_button" else "undo"
                ),
            )
            await session.commit()
        except AuthorizationChanged:
            await session.rollback()
            from financial_dashboard.services.assistant.audit import (
                mark_authorization_changed,
            )

            await mark_authorization_changed(session, interaction_id)
            await session.commit()
            return
    if trigger == "undo" and result.response.outcome != "error":
        try:
            async with async_session() as session:
                from financial_dashboard.services.assistant.mutations import (
                    refresh_caches_after_commit,
                )

                await refresh_caches_after_commit(session)
        except Exception:
            import logging

            logging.getLogger(__name__).exception(
                "Merchant-rule undo committed but cache refresh failed"
            )
    for delivery_id in delivery_ids:
        await dispatch_saved_delivery(delivery_id)


async def _handle_assistant_callback(update, context, *, trigger: str) -> None:
    from financial_dashboard.db import async_session
    from financial_dashboard.services.assistant.audit import (
        claim_interaction,
        claim_processing,
    )

    query = update.callback_query
    message = query.message
    bot_user = await context.bot.get_me()
    if (
        message is None
        or message.from_user is None
        or message.from_user.id != bot_user.id
    ):
        await query.answer("Invalid button")
        return
    async with async_session() as session:
        interaction, is_new = await claim_interaction(
            session,
            telegram_update_id=str(update.update_id),
            chat_id=message.chat.id,
            message_id=message.message_id,
            reply_to_message_id=None,
            trigger=trigger,
            user_text=query.data,
        )
        await session.commit()
        if not is_new:
            await query.answer("Already handled")
            return
        interaction, worker_token = await claim_processing(session, interaction.id)
        assert interaction is not None
        interaction_id = interaction.id
        await session.commit()
    await _process_callback_interaction(
        interaction_id=interaction_id,
        worker_token=worker_token,
        trigger=trigger,
        callback_data=query.data,
        recipient_chat_id=message.chat.id,
        physical_message_id=message.message_id,
    )
    await query.answer()
