"""Strict, provider-neutral contract for one assistant model step.

The model can describe a result or request one of the small, application-owned
tools below.  Pydantic validation is deliberately strict: provider output is
untrusted input and is never interpreted as prose or partially recovered.
"""

import datetime
import re
from decimal import Decimal
from typing import Annotated, Literal, TypeAlias, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    WithJsonSchema,
    field_validator,
    model_validator,
)
from pydantic.types import StrictBool, StrictFloat, StrictInt, StrictStr

from financial_dashboard.core.dates import IST
from financial_dashboard.services.categorization.slugs import UserCategoryBucket


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SetTextValue(StrictModel):
    op: Literal["set"]
    value: StrictStr = Field(max_length=500)


class SetBoolValue(StrictModel):
    op: Literal["set"]
    value: StrictBool


class ClearValue(StrictModel):
    op: Literal["clear"]


TextPatch: TypeAlias = Annotated[SetTextValue | ClearValue, Field(discriminator="op")]


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


class GetTransaction(StrictModel):
    name: Literal["get_transaction"]
    transaction_id: StrictInt = Field(gt=0)


class ListTransactions(StrictModel):
    name: Literal["list_transactions"]
    transaction_ids: list[StrictInt] | None = Field(default=None, max_length=20)
    account_id: StrictInt | None = Field(default=None, gt=0)
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
    limit: StrictInt = Field(default=20, ge=1, le=20)


class ListCategories(StrictModel):
    name: Literal["list_categories"]


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
ReadCall = GetTransaction | ListTransactions | ListCategories | ListReceivables

ToolCall: TypeAlias = Annotated[
    GetTransaction
    | ListTransactions
    | ListCategories
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
    outcome: Literal["answer"]
    text: StrictStr = Field(min_length=1, max_length=4000)


class Clarification(StrictModel):
    outcome: Literal["clarification"]
    question: StrictStr = Field(min_length=1, max_length=1000)
    pending_confirmation: "PendingConfirmation | None" = None


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


class AssistantWriteConfirmation(StrictModel):
    kind: Literal["receivable_write"]
    request: AssistantWrite


class ScreenshotTransaction(StrictModel):
    """What the vision model reads off one UPI app screenshot.

    The model output is untrusted. Code checks every field before a row is
    proposed.
    """

    status: Literal["success", "pending", "failed", "unknown"]
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
    app: Literal["gpay", "phonepe", "paytm", "bhim", "other"]
    account_hint: StrictStr | None = Field(max_length=100)

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
        return re.sub(r"\s", "", value) if value is not None else None

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
    ``duplicate_of`` that the question showed.
    """

    kind: Literal["screenshot_transaction"]
    read: ScreenshotTransaction
    attachment_path: StrictStr = Field(min_length=1, max_length=200)
    media_type: StrictStr = Field(min_length=1, max_length=50)
    missing: list[StrictStr]
    duplicate_of: StrictInt | None = None
    duplicate_voided: StrictBool = False
    not_duplicates: list[StrictInt] = Field(default_factory=list, max_length=20)


ConfirmationAction: TypeAlias = Annotated[
    MerchantRuleConfirmation
    | TransactionChangeConfirmation
    | AssistantWriteConfirmation
    | ScreenshotTransactionConfirmation,
    Field(discriminator="kind"),
]


class StoredConfirmation(StrictModel):
    """The pending confirmation that a conversation stores as JSON."""

    action: ConfirmationAction
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
    """Application-owned reads or one complete atomic mutation request."""

    outcome: Literal["tool_calls"]
    calls: list[ToolCall] = Field(min_length=1, max_length=4)
    explanation: StrictStr = Field(default="", max_length=500)


class Error(StrictModel):
    outcome: Literal["error"]
    message: StrictStr = Field(min_length=1, max_length=1000)
    code: StrictStr = Field(
        default="invalid_model_output", min_length=1, max_length=100
    )


AssistantResponse: TypeAlias = Annotated[
    Answer | Clarification | CategoryProposal | ToolCalls | Error,
    Field(discriminator="outcome"),
]

_RESPONSE_ADAPTER = TypeAdapter(AssistantResponse)


class AssistantResponseEnvelope(StrictModel):
    response: AssistantResponse


def _openai_strict_schema(value: object) -> object:
    """Make every object field required, using nullable unions for optionals."""
    if isinstance(value, list):
        return [_openai_strict_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result: dict[str, object] = {}
    for key, item in value.items():
        if key in {"default", "discriminator"}:
            continue
        output_key = "anyOf" if key == "oneOf" else key
        result[output_key] = _openai_strict_schema(item)
    properties = result.get("properties")
    if result.get("type") == "object" and isinstance(properties, dict):
        result["additionalProperties"] = False
        result["required"] = list(properties)
    return result


def response_json_schema() -> dict[str, object]:
    """Return an object-root schema accepted by strict provider APIs."""
    return cast(
        dict[str, object],
        _openai_strict_schema(AssistantResponseEnvelope.model_json_schema()),
    )


def screenshot_json_schema() -> dict[str, object]:
    """Return the strict provider schema of a screenshot read."""
    return cast(
        dict[str, object],
        _openai_strict_schema(ScreenshotTransaction.model_json_schema()),
    )


def parse_response(data: object) -> AssistantResponse:
    """Validate an already decoded provider object; never parse prose."""

    return _RESPONSE_ADAPTER.validate_python(data)
