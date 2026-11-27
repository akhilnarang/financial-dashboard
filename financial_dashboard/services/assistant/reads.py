"""The read tools of the assistant, shared by the Telegram turn and MCP.

Each read returns the redacted rows that a model sees, never database rows.
"""

import datetime
from decimal import Decimal, InvalidOperation

from sqlalchemy.ext.asyncio import AsyncSession

from financial_dashboard.db.enums import ReceivableStatus
from financial_dashboard.schemas.receivables import ReceivableRead
from financial_dashboard.services.assistant.contracts import (
    ListAccounts,
    ListReceivables,
    ListTransactions,
)
from financial_dashboard.services.assistant.mutations import MutationRejected
from financial_dashboard.services.assistant.transaction_reads import (
    AssistantAccount,
    AssistantTransaction,
    list_accounts,
    list_transactions,
)
from financial_dashboard.services.person_identifiers import normalize_person
from financial_dashboard.services.receivables import list_receivables, receivable_read


async def dispatch_read(
    session: AsyncSession, call: ListAccounts | ListReceivables
) -> list[AssistantAccount] | list[ReceivableRead]:
    """Run one read tool call other than list_transactions.

    Args:
        session: The turn session.
        call: The read call that the model asked for.

    Returns:
        The rows that the call reads.
    """
    match call:
        case ListAccounts():
            return await list_accounts(session)
        case ListReceivables():
            status = ReceivableStatus(call.status) if call.status else None
            person = normalize_person(call.person or "")
            return [
                read
                for read in map(
                    receivable_read, await list_receivables(session, status)
                )
                if (call.kind is None or read.kind == call.kind)
                and person in normalize_person(read.person)
            ]


async def read_transactions(
    session: AsyncSession, call: ListTransactions
) -> list[AssistantTransaction]:
    """Run one list_transactions call.

    Args:
        session: The turn session.
        call: The read call that the model asked for.

    Returns:
        The rows that match the filters.

    Raises:
        MutationRejected: A date or the amount is not valid.
    """
    try:
        date_from = (
            datetime.date.fromisoformat(call.date_from) if call.date_from else None
        )
        date_to = datetime.date.fromisoformat(call.date_to) if call.date_to else None
        amount = Decimal(call.amount) if call.amount else None
    except (ValueError, InvalidOperation) as exc:
        raise MutationRejected("invalid transaction filter") from exc
    if amount is not None and not amount.is_finite():
        raise MutationRejected("invalid transaction filter")
    return await list_transactions(
        session,
        limit=call.limit,
        offset=0,
        account_id=call.account_id,
        card_id=call.card_id,
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
