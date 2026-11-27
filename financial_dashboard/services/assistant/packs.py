"""Tool packs: groups of assistant tools with the prompt lines that they need.

Each step sends the core tools and the loaded packs only. The server picks the
packs before the first step from fixed signals. The model can ask for a pack
that it does not have, and the next step loads it.
"""

from collections.abc import Iterable
from typing import NamedTuple, get_args

from financial_dashboard.services.assistant.contracts import (
    AssistantResponse,
    AddPersonIdentifier,
    ApplyTransactionChanges,
    CreateCategory,
    CreateReceivable,
    GetCardDues,
    GetCashflowSummary,
    GetHoldings,
    GetInvestmentFlows,
    GetNetworth,
    LinkReceivable,
    ListAccounts,
    ListReceivables,
    ListTransactions,
    MergeTransactions,
    PackName,
    ShowTransactions,
    StrictModel,
    ToolCalls,
    UpdateReceivable,
    VoidTransaction,
    JsonSchema,
    response_json_schema,
)
from financial_dashboard.services.assistant.intent_policy import intent_signals

PACK_NAMES: tuple[PackName, ...] = get_args(PackName)
CORE_TOOLS: tuple[type[StrictModel], ...] = (ListTransactions, ListAccounts)
ALL_PACKS: frozenset[PackName] = frozenset(PACK_NAMES)
EDIT_RULES = """\
APPLY is {"name":"apply_transaction_changes","transaction_id":1,
"changes":{optional note/category/exclude_from_cashflow patches},
optional "merchant_rule":{"category":"...","intent_evidence":"..."}}.
A note patch is {"op":"set","value":"..."}, {"op":"append","value":"..."} or {"op":"clear"}.
A category patch is {"op":"set","value":"slug"} or
{"op":"clear","value":null}. An exclusion patch is
{"op":"set","value":true|false}. Omitted fields stay unchanged.
A WRITE can also be {"name":"void_transaction","transaction_id":1}
or {"name":"merge_transactions","keep_id":1,"duplicate_id":2}.
Void a debit when the user says the payment failed, was reversed, or
never happened. Merge when the user says a row is the same payment as
another row: keep the bank row and give the other as duplicate_id.
Find the bank row with list_transactions first. The application asks
the user to confirm a void or a merge.
APPLY uses only an existing category slug.
Act; do not ask what to do. A reply to a transaction that only gives
context, such as 'Snickers in train', is the note: set the note to the
whole message and set your best category. The application asks the user
to confirm a category that the message does not name.
When the TRUSTED TRANSACTION TARGET already has a note, a reply that adds
context appends: use {"op":"append","value":...} with only the new text.
The application keeps the old note and adds the new text after it,
separated by " · ". Use {"op":"set"} on an existing note only when the
user asks to replace or change the note.
A reply that is only a category name sets the category and not the note.
A category_proposal never saves a note. To save a note, use APPLY.
When the user names another active slug, use APPLY.
Change exclude_from_cashflow only when the user asks for it.
The TRUSTED TRANSACTION TARGET is the transaction to change. A number
inside context, such as 'Cashback received for 8864', is part of the
note, not a new target. A transaction number, such as 8788 or #8788, is a
target only when there is no trusted target, or when the user tells you
to change that transaction. Each APPLY changes one transaction.
To change a row that the user describes but does not name, find it
with list_transactions first. Filter by amount, direction, account_id
and a date range. A bank can store only the start of a name, so use a
name only as a search hint to narrow the rows. When the user says
recent or latest, choose the newest match. Wait for the tool result.
Then APPLY to the returned row together with the other changes. Do not ask
in prose whether to change it. The application shows every row and
change, and one Yes applies them all.
For a durable merchant rule, include an exact substring
from the current user message as intent_evidence.
Merchant-rule pattern and priority are server-derived; never provide them."""
RECEIVABLE_RULES = """\
Receivables track money a person owes the user (kind receivable) or
money the user owes a person (kind payable). READ also includes
list_receivables with optional status/kind/person. A RECEIVABLE WRITE is
{"name":"create_receivable","kind":"receivable","person":"...",
"amount":"500.00","note":null,"charge_transaction_ids":[]},
{"name":"link_receivable","receivable_id":1,"transaction_id":1,
"role":"charge|payment"}, {"name":"update_receivable","receivable_id":1,
"amount":null,"status":null,"person":null,"note":null} (null keeps a
field), or {"name":"add_person_identifier","person":"...",
"kind":"counterparty|vpa|phone","value":"..."}.
Use a RECEIVABLE WRITE only when the user asks for that change. The
application asks the user to confirm it. A partial payment is a payment
link. Settle or write off with update_receivable status. Find ids with
list_receivables and list_transactions first."""
REPORT_RULES = """\
A REPORT sends server figures straight to the user. You do not see
them. Make one REPORT call, alone in calls. The REPORTS are:
- {"name":"get_cashflow_summary","date_from":"YYYY-MM-DD",
"date_to":"YYYY-MM-DD"}: income, spending, investment and cashflow
totals over an inclusive range.
- {"name":"get_investment_flows","date_from":"YYYY-MM-DD",
"date_to":"YYYY-MM-DD"}: money put into and taken out of
investments over a range, by counterparty.
- {"name":"get_holdings","search":null,"asset_class":null}: current
funds and stocks with value and units. search matches part of a
holding name. asset_class is one of equity, etf, mutual_fund, bond,
debenture, preference_share, aif, government_security, sgb, other.
- {"name":"get_networth"}: net worth, assets, liabilities and the
latest balance of every account. Use it for account balances.
- {"name":"get_card_dues"}: the latest bill, due date and payment
status of each credit card.
- {"name":"show_transactions", the list_transactions filters except
transaction_ids and limit, "sort":"date|amount|bank|category|counterparty",
"order":"desc|asc"}: the matching rows, their count and totals, and a
link to the same list on the dashboard. Use it when the user asks to
see, list or show transactions. Put an account or card name in bank.
search finds text in a row, such as a merchant. Use list_transactions
only to find rows to change."""
CATEGORY_RULES = """\
A WRITE can also be {"name":"create_category","slug":"...",
"bucket":"expense"}. Use it only when the user asks for a new
category. Set "bucket":"income" only when the user calls it income. The
application asks the user to confirm it."""


class ToolPack(NamedTuple):
    """A group of tools that the server loads together.

    Attributes:
        tools: The tool calls of the pack.
        summary: The index line that the model sees while the pack is not
            loaded.
        instructions: The prompt rules that only these tools need.
    """

    tools: tuple[type[StrictModel], ...]
    summary: str
    instructions: str


class PackChoice(NamedTuple):
    """The packs of one step, with the packs that the step can ask for.

    Attributes:
        loaded: The packs whose tools and rules the step has.
        requestable: The other packs, in index order.
    """

    loaded: frozenset[PackName]
    requestable: tuple[PackName, ...]


PACKS: dict[PackName, ToolPack] = {
    "edit": ToolPack(
        tools=(ApplyTransactionChanges, VoidTransaction, MergeTransactions),
        summary=(
            "change the note, category or cashflow exclusion of a transaction, "
            "save a merchant rule, void a failed payment, or merge a duplicate row."
        ),
        instructions=EDIT_RULES,
    ),
    "receivables": ToolPack(
        tools=(
            ListReceivables,
            CreateReceivable,
            LinkReceivable,
            UpdateReceivable,
            AddPersonIdentifier,
        ),
        summary=(
            "read, create, link, settle or change money that a person owes the "
            "user or that the user owes a person, and save a UPI id or phone."
        ),
        instructions=RECEIVABLE_RULES,
    ),
    "reports": ToolPack(
        tools=(
            GetCashflowSummary,
            GetInvestmentFlows,
            GetHoldings,
            GetNetworth,
            GetCardDues,
            ShowTransactions,
        ),
        summary=(
            "server figures for cashflow, investment flows, holdings, net worth, "
            "account balances and card dues, and a list of transactions to show."
        ),
        instructions=REPORT_RULES,
    ),
    "categories": ToolPack(
        tools=(CreateCategory,),
        summary="create a new category.",
        instructions=CATEGORY_RULES,
    ),
}


def pack_choice(loaded: Iterable[PackName]) -> PackChoice:
    """Name the loaded packs of a step and the packs that it can ask for.

    Args:
        loaded: The packs that the step has.

    Returns:
        The loaded packs, and the others in index order.
    """
    packs = frozenset(loaded)
    return PackChoice(packs, tuple(name for name in PACK_NAMES if name not in packs))


def loaded_tool_names(packs: frozenset[PackName]) -> frozenset[str]:
    """Name every tool that a step with these packs can call.

    Args:
        packs: The loaded packs.

    Returns:
        The tool names of the core tools and of the loaded packs.
    """
    return frozenset(tool_name(tool) for tool in _loaded_tools(packs))


def tool_name(tool: type[StrictModel]) -> str:
    """Return the name that the model gives to call this tool.

    Args:
        tool: A tool call model.

    Returns:
        The value of its ``name`` literal, such as ``list_accounts``.
    """
    return get_args(tool.model_fields["name"].annotation)[0]


def pack_response_schema(packs: frozenset[PackName]) -> JsonSchema:
    """Build the strict response schema of a step with these packs.

    Args:
        packs: The loaded packs.

    Returns:
        The schema, with the core tools, the tools of the loaded packs, and a
        ``request_packs`` outcome for the other packs.
    """
    return response_json_schema(
        loaded_tool_names(packs), pack_choice(packs).requestable
    )


def pack_refusal(response: AssistantResponse, packs: frozenset[PackName]) -> str | None:
    """Say why a reply calls a tool outside its step's packs, or return None.

    The model must never use a tool whose rules it did not see.

    Args:
        response: The validated reply of the step.
        packs: The loaded packs.

    Returns:
        The audit detail of the refusal, or None when the reply keeps to the
        packs.
    """
    if not isinstance(response, ToolCalls):
        return None
    unloaded = {call.name for call in response.calls} - loaded_tool_names(packs)
    return f"tool not loaded: {', '.join(sorted(unloaded))}" if unloaded else None


def choose_packs(user_texts: Iterable[str], *, has_target: bool) -> frozenset[PackName]:
    """Pick the packs of a turn before the first model call, with no model.

    A reply to a transaction always loads edit. A turn with no trusted target
    is an ``/ask`` or a reply in its conversation, and loads reports. Fixed
    signals in the user text add the other packs.

    Args:
        user_texts: The current user message and the user turns before it.
        has_target: Whether the turn has a trusted transaction target.

    Returns:
        The packs to load.
    """
    packs: set[PackName] = {"edit"} if has_target else {"reports"}
    for text in user_texts:
        signals = intent_signals(text)
        if signals.edit:
            packs.add("edit")
        if signals.receivables:
            packs.add("receivables")
        if signals.reports:
            packs.add("reports")
        if signals.categories:
            packs.add("categories")
    return frozenset(packs)


def _loaded_tools(packs: frozenset[PackName]) -> tuple[type[StrictModel], ...]:
    """List the core tools, then the tools of the loaded packs in index order."""
    return CORE_TOOLS + tuple(
        tool for name in PACK_NAMES if name in packs for tool in PACKS[name].tools
    )
