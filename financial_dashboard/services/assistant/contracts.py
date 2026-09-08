"""Strict, provider-neutral contract for one assistant model step.

The model can describe a result or request one of the small, application-owned
tools below.  Pydantic validation is deliberately strict: provider output is
untrusted input and is never interpreted as prose or partially recovered.
"""

from typing import Annotated, Literal, TypeAlias, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)
from pydantic.types import StrictBool, StrictFloat, StrictInt, StrictStr


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


ToolCall: TypeAlias = Annotated[
    GetTransaction | ListTransactions | ListCategories | ApplyTransactionChanges,
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


class CategoryCandidate(StrictModel):
    slug: StrictStr = Field(min_length=1, max_length=100)
    reason: StrictStr = Field(min_length=1, max_length=300)
    confidence: StrictFloat = Field(ge=0, le=1)


class CategoryProposal(StrictModel):
    outcome: Literal["category_proposal"]
    transaction_id: StrictInt = Field(gt=0)
    explanation: StrictStr = Field(min_length=1, max_length=500)
    candidates: list[CategoryCandidate] = Field(min_length=2, max_length=3)

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


def parse_response(data: object) -> AssistantResponse:
    """Validate an already decoded provider object; never parse prose."""

    return _RESPONSE_ADAPTER.validate_python(data)
