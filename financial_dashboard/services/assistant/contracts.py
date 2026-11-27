"""Strict, provider-neutral contract for one assistant model step.

The model can describe a result or request one of the small, application-owned
tools below.  Pydantic validation is deliberately strict: provider output is
untrusted input and is never interpreted as prose or partially recovered.
"""

import copy
import datetime
import re
from collections.abc import Sequence
from decimal import Decimal
from typing import Annotated, Literal, TypeAlias, TypedDict, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    WithJsonSchema,
    field_validator,
    model_validator,
)
from pydantic.types import StrictBool, StrictFloat, StrictInt, StrictStr

from financial_dashboard.core.dates import IST
from financial_dashboard.services.categorization.slugs import UserCategoryBucket


PackName: TypeAlias = Literal["edit", "receivables", "reports", "categories"]
# A JSON schema. The keys are the keywords that the strict schema code reads or
# writes. A schema can also hold other keywords, such as title or minLength.
JsonSchema = TypedDict(
    "JsonSchema",
    {
        "$defs": dict[str, "JsonSchema"],
        "$ref": str,
        "additionalProperties": bool,
        "anyOf": list["JsonSchema"],
        "const": str,
        "default": JsonValue,
        "description": str,
        "discriminator": JsonValue,
        "enum": Sequence[str],
        "items": "JsonSchema",
        "maxItems": int,
        "oneOf": list["JsonSchema"],
        "properties": dict[str, "JsonSchema"],
        "required": list[str],
    },
    total=False,
    extra_items=JsonValue,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SetTextValue(StrictModel):
    op: Literal["set"]
    value: StrictStr = Field(max_length=500)


class SetBoolValue(StrictModel):
    op: Literal["set"]
    value: StrictBool


class AppendTextValue(StrictModel):
    op: Literal["append"]
    value: StrictStr = Field(max_length=500)


class ClearValue(StrictModel):
    op: Literal["clear"]


TextPatch: TypeAlias = Annotated[
    SetTextValue | AppendTextValue | ClearValue, Field(discriminator="op")
]


class CategoryPatch(StrictModel):
    """A category patch; category values are slugs, validated server-side."""

    op: Literal["set", "clear"]
    value: StrictStr | None = Field(default=None, min_length=1, max_length=100)

    @field_validator("value")
    @classmethod
    def normalize_slug(cls, value: str | None) -> str | None:
        return value.lower().strip() if value is not None else None

    @model_validator(mode="after")
    def validate_operation(self) -> "CategoryPatch":
        if self.op == "set" and self.value is None:
            raise ValueError("category set requires a value")
        if self.op == "clear" and self.value is not None:
            raise ValueError("category clear cannot include a value")
        return self


class TransactionChanges(StrictModel):
    """Explicit patch semantics: omitted fields remain unchanged."""

    note: TextPatch | None = None
    category: CategoryPatch | None = None
    exclude_from_cashflow: SetBoolValue | None = None


class MerchantRuleRequest(StrictModel):
    category: StrictStr = Field(min_length=1, max_length=100)
    intent_evidence: StrictStr = Field(min_length=1, max_length=240)

    @field_validator("category")
    @classmethod
    def normalize_category(cls, value: str) -> str:
        return value.lower().strip()


class ApplyTransactionChanges(StrictModel):
    name: Literal["apply_transaction_changes"]
    transaction_id: StrictInt = Field(gt=0)
    changes: TransactionChanges
    merchant_rule: MerchantRuleRequest | None = None


class TransactionFilterFields(StrictModel):
    """The transaction filters that a read and the transaction report share.

    Every field is optional. The model gives dates and the amount as text, and
    the server checks them.

    Attributes:
        account_id: One account, by id.
        card_id: One card, by id.
        date_from: The first date, as YYYY-MM-DD.
        date_to: The last date, as YYYY-MM-DD.
        direction: ``debit`` or ``credit``.
        amount: The exact amount, as text.
        bank: A bank, or the name of an account or a card.
        source: The row source, such as ``sms``.
        category: A category slug, or the label that the user sees.
        review_status: The category review status.
        reference: A payment reference, such as a UTR.
        search: Text to find in the row, such as a merchant.
        excluded: True for rows out of the cashflow, False for rows in it.
    """

    account_id: StrictInt | None = Field(default=None, gt=0)
    card_id: StrictInt | None = Field(default=None, gt=0)
    date_from: StrictStr | None = Field(default=None, max_length=30)
    date_to: StrictStr | None = Field(default=None, max_length=30)
    direction: Literal["debit", "credit"] | None = None
    amount: StrictStr | None = Field(default=None, max_length=40)
    bank: StrictStr | None = Field(default=None, max_length=100)
    source: StrictStr | None = Field(default=None, max_length=100)
    category: StrictStr | None = Field(default=None, max_length=100)
    review_status: StrictStr | None = Field(default=None, max_length=40)
    reference: StrictStr | None = Field(default=None, max_length=200)
    search: StrictStr | None = Field(default=None, max_length=200)
    excluded: StrictBool | None = None


class ListTransactions(TransactionFilterFields):
    """Ask for transaction rows that the model reads, to find or explain them.

    Attributes:
        name: The tool name.
        transaction_ids: Read these rows by id.
        limit: The most rows to return.
    """

    name: Literal["list_transactions"]
    transaction_ids: list[StrictInt] | None = Field(default=None, max_length=20)
    limit: StrictInt = Field(default=20, ge=1, le=20)


class ShowTransactions(TransactionFilterFields):
    """Ask the server to send the filtered rows, with a link to their page.

    The rows, the totals and the link go to the user. The model does not see
    them.

    Attributes:
        name: The tool name.
        sort: The column to sort by, as the transactions page names it.
        order: The sort order.
    """

    name: Literal["show_transactions"]
    sort: Literal["amount", "bank", "category", "counterparty", "date"] = "date"
    order: Literal["asc", "desc"] = "desc"


class GetCashflowSummary(StrictModel):
    """Ask the server for the cashflow figures of one inclusive date range."""

    name: Literal["get_cashflow_summary"]
    date_from: StrictStr = Field(max_length=30)
    date_to: StrictStr = Field(max_length=30)


class GetInvestmentFlows(StrictModel):
    """Ask the server for investment money in and out of one date range."""

    name: Literal["get_investment_flows"]
    date_from: StrictStr = Field(max_length=30)
    date_to: StrictStr = Field(max_length=30)


class GetHoldings(StrictModel):
    """Ask the server for current holdings, optionally filtered."""

    name: Literal["get_holdings"]
    search: StrictStr | None = Field(default=None, max_length=100)
    asset_class: (
        Literal[
            "equity",
            "preference_share",
            "bond",
            "debenture",
            "etf",
            "mutual_fund",
            "aif",
            "government_security",
            "sgb",
            "other",
        ]
        | None
    ) = None


class GetNetworth(StrictModel):
    """Ask the server for net worth and every account balance."""

    name: Literal["get_networth"]


class GetCardDues(StrictModel):
    """Ask the server for the latest bill of each credit card."""

    name: Literal["get_card_dues"]


class ListAccounts(StrictModel):
    """Ask for the user's accounts and cards, with their ids and names.

    Attributes:
        name: The tool name.
    """

    name: Literal["list_accounts"]


ReceivableKindName: TypeAlias = Literal["receivable", "payable"]
ReceivableStatusName: TypeAlias = Literal["open", "settled", "written_off"]
PersonName: TypeAlias = Annotated[StrictStr, Field(min_length=1, max_length=200)]
MoneyText: TypeAlias = Annotated[StrictStr, Field(min_length=1, max_length=40)]


class ListReceivables(StrictModel):
    name: Literal["list_receivables"]
    status: ReceivableStatusName | None = None
    kind: ReceivableKindName | None = None
    person: StrictStr | None = Field(default=None, max_length=200)


class CreateReceivable(StrictModel):
    name: Literal["create_receivable"]
    kind: ReceivableKindName
    person: PersonName
    amount: MoneyText
    note: StrictStr | None = Field(default=None, max_length=500)
    charge_transaction_ids: list[StrictInt] = Field(default_factory=list, max_length=20)


class LinkReceivable(StrictModel):
    name: Literal["link_receivable"]
    receivable_id: StrictInt = Field(gt=0)
    transaction_id: StrictInt = Field(gt=0)
    role: Literal["charge", "payment"]


class UpdateReceivable(StrictModel):
    """Omitted or null fields stay unchanged."""

    name: Literal["update_receivable"]
    receivable_id: StrictInt = Field(gt=0)
    amount: MoneyText | None = None
    status: ReceivableStatusName | None = None
    person: PersonName | None = None
    note: StrictStr | None = Field(default=None, max_length=500)


class AddPersonIdentifier(StrictModel):
    name: Literal["add_person_identifier"]
    person: PersonName
    kind: Literal["counterparty", "vpa", "phone"]
    value: StrictStr = Field(min_length=1, max_length=200)


class CreateCategory(StrictModel):
    name: Literal["create_category"]
    slug: StrictStr = Field(min_length=1, max_length=100)
    bucket: UserCategoryBucket = "expense"


class VoidTransaction(StrictModel):
    name: Literal["void_transaction"]
    transaction_id: StrictInt = Field(gt=0)


class MergeTransactions(StrictModel):
    name: Literal["merge_transactions"]
    keep_id: StrictInt = Field(gt=0)
    duplicate_id: StrictInt = Field(gt=0)


AssistantWriteCall = (
    CreateReceivable
    | LinkReceivable
    | UpdateReceivable
    | AddPersonIdentifier
    | CreateCategory
    | VoidTransaction
    | MergeTransactions
)
AssistantWrite: TypeAlias = Annotated[AssistantWriteCall, Field(discriminator="name")]
ReportCall = (
    GetCashflowSummary
    | GetInvestmentFlows
    | GetHoldings
    | GetNetworth
    | GetCardDues
    | ShowTransactions
)

ToolCall: TypeAlias = Annotated[
    ListTransactions
    | ListAccounts
    | GetCashflowSummary
    | GetInvestmentFlows
    | GetHoldings
    | GetNetworth
    | GetCardDues
    | ShowTransactions
    | ApplyTransactionChanges
    | ListReceivables
    | CreateReceivable
    | LinkReceivable
    | UpdateReceivable
    | AddPersonIdentifier
    | CreateCategory
    | VoidTransaction
    | MergeTransactions,
    Field(discriminator="name"),
]


class Answer(StrictModel):
    """A text reply to the user.

    Attributes:
        outcome: The outcome kind.
        text: The reply text.
        transaction_ids: The rows that the text tells the user about. The
            application sends a card for each row that a read of this turn
            returned.
    """

    outcome: Literal["answer"]
    text: StrictStr = Field(min_length=1, max_length=4000)
    transaction_ids: list[StrictInt] = Field(default_factory=list, max_length=20)


class Clarification(StrictModel):
    """A question back to the user.

    Attributes:
        outcome: The outcome kind.
        question: The question text.
        pending_confirmation: The change that a yes confirms, or None.
        transaction_ids: The rows that the question tells the user about.
            The application sends a card for each row that a read of this
            turn returned.
    """

    outcome: Literal["clarification"]
    question: StrictStr = Field(min_length=1, max_length=1000)
    pending_confirmation: "PendingConfirmation | None" = None
    transaction_ids: list[StrictInt] = Field(default_factory=list, max_length=20)


class MerchantRuleConfirmation(StrictModel):
    kind: Literal["merchant_rule"]
    transaction_id: StrictInt = Field(gt=0)
    category: StrictStr = Field(min_length=1, max_length=100)

    @field_validator("category")
    @classmethod
    def normalize_category(cls, value: str) -> str:
        return value.lower().strip()


PendingConfirmation: TypeAlias = MerchantRuleConfirmation


class TransactionChangeConfirmation(StrictModel):
    kind: Literal["transaction_change"]
    transaction_id: StrictInt = Field(gt=0)
    request: ApplyTransactionChanges


class TransactionChangesConfirmation(StrictModel):
    """Changes to rows that a search found, which one Yes applies together.

    All the changes apply, or none of them. When ``candidates`` is not empty,
    the request at index ``choose`` has no fixed row. The user taps one
    candidate, and that row gets the change.
    """

    kind: Literal["transaction_changes"]
    requests: list[ApplyTransactionChanges] = Field(min_length=1, max_length=4)
    choose: StrictInt | None = Field(default=None, ge=0)
    candidates: list[StrictInt] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def validate_choice(self) -> "TransactionChangesConfirmation":
        if (self.choose is None) != (not self.candidates):
            raise ValueError("a choice needs candidates, and candidates a choice")
        if self.choose is not None and self.choose >= len(self.requests):
            raise ValueError("the choice names no request")
        return self


class AssistantWriteConfirmation(StrictModel):
    kind: Literal["receivable_write"]
    request: AssistantWrite


class UpiParty(StrictModel):
    """One party, From or To, on a UPI receipt."""

    name: StrictStr | None = Field(default=None, max_length=200)
    vpa: StrictStr | None = Field(default=None, max_length=200)
    bank_text: StrictStr | None = Field(default=None, max_length=100)


class ScreenshotTransaction(StrictModel):
    """What the vision model reads off one payment screenshot.

    The screenshot is a UPI app screen, a bank transfer receipt or a card app
    screen. The model output is untrusted. Code checks every field before a
    row is proposed.

    ``status`` holds ``reversed`` when a refund or a reversal returned money.
    A reversed read always has ``direction`` of ``credit``. ``payer`` and
    ``payee`` hold the From and To blocks of a receipt.
    """

    status: Literal["success", "pending", "failed", "reversed", "unknown"]
    direction: Literal["debit", "credit"]
    amount: (
        Annotated[
            Decimal,
            WithJsonSchema({"type": "string", "description": "Rupees, e.g. 950.00"}),
        ]
        | None
    )
    counterparty: StrictStr | None = Field(max_length=200)
    vpa: StrictStr | None = Field(max_length=200)
    utr: StrictStr | None = Field(max_length=40)
    occurred_at: (
        Annotated[
            datetime.datetime,
            WithJsonSchema(
                {"type": "string", "description": "IST as shown, YYYY-MM-DDTHH:MM:SS"}
            ),
        ]
        | None
    )
    app: Literal[
        "gpay",
        "phonepe",
        "paytm",
        "bhim",
        "cred",
        "supermoney",
        "bank",
        "other",
    ]
    channel: Literal["upi", "imps", "neft", "rtgs", "card"] = "upi"
    account_hint: StrictStr | None = Field(max_length=100)
    payer: UpiParty | None = None
    payee: UpiParty | None = None

    @model_validator(mode="after")
    def reversed_is_a_credit(self) -> ScreenshotTransaction:
        """Set the direction of a reversed read, and drop the UTR of a card read.

        A refund or a reversal returns money to the owner. The model can read
        the direction of the original payment. The status decides the money
        flow, so the read always records a credit.

        A UTR is a UPI reference. A card read has none, and a stored reference
        would match or collide with rows of another payment.

        Returns:
            The read, with ``direction`` set to ``credit`` when reversed and
            ``utr`` set to None for a card read.
        """
        if self.status == "reversed":
            self.direction = "credit"
        if self.channel == "card":
            self.utr = None
        return self

    @field_validator("amount")
    @classmethod
    def storable_amount(cls, value: Decimal | None) -> Decimal | None:
        """Drop an amount the amount column cannot hold, so the checks ask for it.

        The column holds 10 integer digits and 2 decimals. The bound comes
        first, because quantize fails on a huge value.
        """
        if (
            value is None
            or not Decimal(0) < value < Decimal("1e10")
            or value != value.quantize(Decimal("0.01"))
        ):
            return None
        return value

    @field_validator("utr")
    @classmethod
    def strip_utr_spaces(cls, value: str | None) -> str | None:
        """Drop spaces and uppercase the reference, as banks print a UTR."""
        return re.sub(r"\s", "", value).upper() if value is not None else None

    @field_validator("occurred_at")
    @classmethod
    def as_ist(cls, value: datetime.datetime | None) -> datetime.datetime | None:
        if value is None or value.tzinfo is None:
            return value
        return value.astimezone(IST).replace(tzinfo=None)


class ScreenshotTransactionConfirmation(StrictModel):
    """A screenshot read that a Yes tap saves, or attaches to ``duplicate_of``.

    ``missing`` names the mandatory fields that failed the checks. A Yes
    never saves while it is not empty. ``not_duplicates`` holds the rows the
    owner called another payment. ``duplicate_voided`` is the void state of
    ``duplicate_of`` that the question showed. ``notes`` holds the caption and
    the replies, oldest first, so each new read sees all of them.
    ``direction_settled`` is False while the receipt blocks do not show which
    side is the owner and the owner has not said received or paid. A Yes
    never saves while it is False. ``owner_direction`` is what the owner said,
    and each new read keeps it. ``self_transfer`` is True when the money moved
    between the owner's own accounts, so each review records the missing leg.
    """

    kind: Literal["screenshot_transaction"]
    read: ScreenshotTransaction
    attachment_path: StrictStr = Field(min_length=1, max_length=200)
    media_type: StrictStr = Field(min_length=1, max_length=50)
    missing: list[StrictStr]
    duplicate_of: StrictInt | None = None
    duplicate_voided: StrictBool = False
    not_duplicates: list[StrictInt] = Field(default_factory=list, max_length=20)
    notes: list[StrictStr] = Field(default_factory=list, max_length=20)
    direction_settled: StrictBool = True
    owner_direction: Literal["credit", "debit"] | None = None
    self_transfer: StrictBool = False


ConfirmationAction: TypeAlias = Annotated[
    MerchantRuleConfirmation
    | TransactionChangeConfirmation
    | TransactionChangesConfirmation
    | AssistantWriteConfirmation
    | ScreenshotTransactionConfirmation,
    Field(discriminator="kind"),
]


ChainedWrite: TypeAlias = Annotated[
    ApplyTransactionChanges | AssistantWriteCall, Field(discriminator="name")
]


class PendingChain(StrictModel):
    """The writes that follow the pending one, from the same user message.

    After a Yes applies the pending write, the next write is checked and asked
    for in turn. ``transaction_id`` is the trusted target of that message.
    """

    calls: list[ChainedWrite] = Field(min_length=1, max_length=4)
    message: StrictStr = Field(min_length=1)
    transaction_id: StrictInt | None = None


class StoredConfirmation(StrictModel):
    """The pending confirmation that a conversation stores as JSON."""

    action: ConfirmationAction
    chain: PendingChain | None = None
    policy_version: StrictStr
    direction_policy: Literal["inferred_strict", "explicit_manual_override"] = (
        "inferred_strict"
    )


class CategoryCandidate(StrictModel):
    slug: StrictStr = Field(min_length=1, max_length=100)
    reason: StrictStr = Field(min_length=1, max_length=300)
    confidence: StrictFloat = Field(ge=0, le=1)


class CategoryProposal(StrictModel):
    outcome: Literal["category_proposal"]
    transaction_id: StrictInt = Field(gt=0)
    explanation: StrictStr = Field(min_length=1, max_length=500)
    # The application offers one guessed category after a shorthand note.
    candidates: list[CategoryCandidate] = Field(min_length=1, max_length=3)

    @field_validator("candidates")
    @classmethod
    def unique_candidates(
        cls, values: list[CategoryCandidate]
    ) -> list[CategoryCandidate]:
        if len({candidate.slug for candidate in values}) != len(values):
            raise ValueError("category candidates must be unique")
        return values


class ToolCalls(StrictModel):
    """Application-owned reads and up to 4 writes.

    The first write applies or asks now. The others form a chain that asks for
    one Yes per write, so the writes are not atomic. When a change targets a
    row that a search found, all the changes ask for one Yes together.
    """

    outcome: Literal["tool_calls"]
    calls: list[ToolCall] = Field(min_length=1, max_length=4)
    explanation: StrictStr = Field(default="", max_length=500)


class Error(StrictModel):
    outcome: Literal["error"]
    message: StrictStr = Field(min_length=1, max_length=1000)
    code: StrictStr = Field(
        default="invalid_model_output", min_length=1, max_length=100
    )


class RequestPacks(StrictModel):
    """Ask the server to load tool packs that this step does not have.

    The next step has the tools and the rules of these packs.

    Attributes:
        outcome: The outcome name.
        packs: The packs to load.
    """

    outcome: Literal["request_packs"]
    packs: list[PackName] = Field(min_length=1, max_length=4)


AssistantResponse: TypeAlias = Annotated[
    Answer | Clarification | CategoryProposal | ToolCalls | Error | RequestPacks,
    Field(discriminator="outcome"),
]

_RESPONSE_ADAPTER = TypeAdapter(AssistantResponse)


class AssistantResponseEnvelope(StrictModel):
    """The object root of every model reply.

    Strict provider APIs need an object at the root, not a union.

    Attributes:
        response: The outcome of the step.
    """

    response: AssistantResponse


def _openai_strict_schema(schema: JsonSchema) -> JsonSchema:
    """Make every object field required, using nullable unions for optionals.

    Strict provider APIs accept no default and no discriminator, and they
    take anyOf, not oneOf. The const property of a union branch, such as a
    tool name, comes before the other properties. A model's description drops the Attributes
    section of its docstring. That section is for the reader of the code,
    and the schema goes with every model call.

    Args:
        schema: A pydantic JSON schema, or one of its sub-schemas.

    Returns:
        A new schema in the strict form. The input does not change.
    """
    strict = schema.copy()
    strict.pop("default", None)
    strict.pop("discriminator", None)
    if (one_of := strict.pop("oneOf", None)) is not None:
        strict["anyOf"] = one_of
    if "anyOf" in strict:
        strict["anyOf"] = [_openai_strict_schema(item) for item in strict["anyOf"]]
    if "items" in strict:
        strict["items"] = _openai_strict_schema(strict["items"])
    if "$defs" in strict:
        strict["$defs"] = {
            name: _openai_strict_schema(body) for name, body in strict["$defs"].items()
        }
    if "description" in strict:
        strict["description"] = strict["description"].split("\n\nAttributes:", 1)[0]
    if "properties" in strict:
        # A strict provider writes the properties in schema order. A tool name
        # after the arguments makes that tool unreachable once the model has
        # written "name" first, so a const property comes first.
        properties = sorted(
            strict["properties"].items(), key=lambda item: "const" not in item[1]
        )
        strict["properties"] = {
            name: _openai_strict_schema(body) for name, body in properties
        }
        strict["additionalProperties"] = False
        strict["required"] = list(strict["properties"])
    return strict


def _model_schema(model: type[BaseModel]) -> JsonSchema:
    """Return the strict provider schema of a pydantic model.

    Args:
        model: The model to describe.

    Returns:
        The model's JSON schema in the strict form.
    """
    # Pydantic types its JSON schema as a plain dict.
    return _openai_strict_schema(cast(JsonSchema, model.model_json_schema()))


_FULL_RESPONSE_SCHEMA = _model_schema(AssistantResponseEnvelope)
# The docstring is for the reader of the code. The model does not need it.
del _FULL_RESPONSE_SCHEMA["description"]


def response_json_schema(
    tool_names: frozenset[str], requestable: Sequence[PackName]
) -> JsonSchema:
    """Return the strict provider schema of one step, for the given tools only.

    The full schema drops the tools that the step cannot call, and the
    definitions that nothing then uses.

    Args:
        tool_names: The names of the tools that the step can call.
        requestable: The packs that the step can ask for. With none, the
            schema has no ``request_packs`` outcome.

    Returns:
        An object-root schema that strict provider APIs accept.
    """
    schema = copy.deepcopy(_FULL_RESPONSE_SCHEMA)
    defs = schema["$defs"]
    calls = defs["ToolCalls"]["properties"]["calls"]["items"]
    calls["anyOf"] = [
        ref
        for ref in calls["anyOf"]
        if defs[_ref_name(ref)]["properties"]["name"]["const"] in tool_names
    ]
    outcomes = schema["properties"]["response"]
    if requestable:
        packs = defs["RequestPacks"]["properties"]["packs"]
        packs["items"]["enum"] = list(requestable)
        packs["maxItems"] = len(requestable)
    else:
        outcomes["anyOf"] = [
            ref for ref in outcomes["anyOf"] if _ref_name(ref) != "RequestPacks"
        ]
    used = _used_defs(schema, defs)
    schema["$defs"] = {name: body for name, body in defs.items() if name in used}
    return schema


def _ref_name(ref: JsonSchema) -> str:
    """Return the definition name of a ``{"$ref": "#/$defs/Name"}`` entry."""
    return ref["$ref"].rsplit("/", 1)[1]


def _used_defs(schema: JsonSchema, defs: dict[str, JsonSchema]) -> set[str]:
    """Name the definitions that the schema root reaches through its refs.

    The walk follows only the keywords that hold sub-schemas. A field name
    in properties is never read as a keyword.

    Args:
        schema: The schema root.
        defs: The definitions that the refs point to.

    Returns:
        The names of the reached definitions.
    """
    used: set[str] = set()
    pending = [schema]
    while pending:
        node = pending.pop()
        if "$ref" in node and (name := _ref_name(node)) not in used:
            used.add(name)
            pending.append(defs[name])
        pending.extend(node.get("properties", {}).values())
        pending.extend(node.get("anyOf", []))
        if "items" in node:
            pending.append(node["items"])
    return used


def screenshot_json_schema() -> JsonSchema:
    """Return the strict provider schema of a screenshot read."""
    return _model_schema(ScreenshotTransaction)


def parse_response(data: object) -> AssistantResponse:
    """Validate an already decoded provider object; never parse prose."""

    return _RESPONSE_ADAPTER.validate_python(data)
