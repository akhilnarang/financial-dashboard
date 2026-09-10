"""Provider-neutral LLM contract for categorization.

Holds the prompt, the result type, and the result parser. Both the Gemini
and the OpenAI-compatible provider import from here. Each provider module
only owns its transport: client construction, the call, and its own
structured-output flag.
"""

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any, NamedTuple

from financial_dashboard.services.categorization.fewshot import FewShotExample
from financial_dashboard.services.categorization.normalize import (
    normalize_text,
    redact_names,
    redact_pii,
)

NEEDS_REVIEW = "needs_review"
LLM_TIMEOUT_MS = (
    30_000  # cap per-call latency so a slow provider can't stall the poll loop
)


class LlmCandidate(NamedTuple):
    slug: str
    confidence: float


class LlmResult(NamedTuple):
    slug: str
    confidence: float
    reason: str
    candidates: tuple[LlmCandidate, ...] = ()


# A short, evidenced note per bank on how to read its raw narration codes.
# Only add a bank here once a real sample shows the model needs the help --
# do not guess a format. Keyed by fields["bank"] (the same slug the parsers
# use), so it lines up with whichever bank produced the raw_description.
BANK_NARRATION_HINTS: dict[str, str] = {
    "indusind": (
        "IndusInd UPI descriptions look like "
        "'UPI/<ref>/DR|CR/<name>/<bank-code>/<vpa>'. DR/CR marks debit or "
        "credit here. It is not part of a name, and it does not mean doctor "
        "or healthcare."
    ),
    "idfc": (
        "IDFC UPI descriptions look like 'UPI/DR|CR/<ref>/<name>/<vpa>/"
        "<bank-code>/<note>'. DR/CR marks debit or credit here. It is not "
        "part of a name, and it does not mean doctor or healthcare."
    ),
    "sbi": (
        "SBI UPI descriptions look like "
        "'UPI/DR|CR/<ref>/<name>/<bank-code>/<vpa>/<remark>'. DR/CR marks "
        "debit or credit here. It is not part of a name, and it does not "
        "mean doctor or healthcare."
    ),
    "uboi": (
        "Union Bank UPI descriptions look like "
        "'UPIAB|UPIAR/<ref>/DR|CR/<name>/<bank-code>/<vpa>'. DR/CR marks "
        "debit or credit here. It is not part of a name, and it does not "
        "mean doctor or healthcare."
    ),
}


def _sanitize_counterparty(text: str | None, name_tokens: Sequence[str]) -> str:
    """Counterparty: strip numbers (PII) then mask configured name tokens."""
    return redact_names(redact_pii(text), name_tokens)


def _sanitize_description(text: str | None, name_tokens: Sequence[str]) -> str:
    """Description: strip numbers, mask names, then normalize whitespace/case."""
    return normalize_text(redact_names(redact_pii(text), name_tokens))


def build_prompt(
    *,
    fields: Mapping[str, str | None],
    examples: Sequence[FewShotExample],
    active_slugs: list[str],
    name_tokens: Sequence[str] = (),
) -> str:
    lines = [
        "You are a personal-finance transaction categorizer.",
        "Choose exactly ONE category slug from this list:",
        ", ".join(active_slugs),
        f'If none fit, return "{NEEDS_REVIEW}".',
        "The unknown category is reserved for empty input handled by the system; "
        "use needs_review when a transaction's purpose is ambiguous.",
        "Return JSON with category, confidence (0..1), reason (one short sentence), "
        "and candidates: an ordered list of zero to three plausible category "
        "objects, each {category, confidence (0..1)}. Include the primary "
        "category when you have one; use an empty list when none fit.",
        "",
        "Classify the transaction purpose using merchant identity and narration; "
        "an itemized receipt is not required for a recognizable merchant. "
        "groceries covers grocery/quick-commerce purchases; dining covers "
        "prepared food and drinks. Confidence measures support for the category, "
        "not whether you know the individual items purchased.",
        "A bank-account credit is money received; a debit is money sent. "
        "Transfers, investments and credit-card payments are not ordinary spending.",
        "On a credit_card account, a credit reduces the card balance: use "
        "credit_card_payment for paying the card bill, refund for a merchant refund, "
        "or cashback_rewards for rewards. It is never salary, interest, "
        "other_income or repayment. A card debit is usually a purchase or charge.",
        "A bank-account credit from an individual paying you back = repayment; "
        "a merchant returning a purchase payment = refund.",
        "Do NOT use self_transfer (handled separately). For money moved to/from another "
        "person, use 'repayment' for a credit or 'expense'/the specific spending category "
        "for a debit only when the purpose supports it. Use family or reimbursement "
        "when the context establishes that purpose. Unclear person-to-person "
        "transfers and payment gateways alone do not establish a spending purpose; "
        "return needs_review when the distinction remains unclear.",
        "",
    ]
    if examples:
        lines.append("Examples of previously categorized transactions:")
        for ex in examples:
            lines.append(
                f"- [{ex.direction}] {_sanitize_counterparty(ex.counterparty, name_tokens)} "
                f"| {_sanitize_description(ex.raw_description, name_tokens)} -> {ex.category}"
            )
        lines.append("")
    lines.append("Transaction to categorize:")
    lines.append(f"bank: {fields.get('bank')}")
    lines.append(f"account_type: {fields.get('account_type')}")
    lines.append(f"email_type: {fields.get('email_type')}")
    lines.append(f"direction: {fields.get('direction')}")
    lines.append(f"amount: {fields.get('amount')} {fields.get('currency')}")
    lines.append(f"channel: {fields.get('channel')}")
    if hint := BANK_NARRATION_HINTS.get(fields.get("bank") or ""):
        lines.append(f"format note: {hint}")
    lines.append(
        f"counterparty: {_sanitize_counterparty(fields.get('counterparty'), name_tokens)}"
    )
    lines.append(
        f"description: {_sanitize_description(fields.get('raw_description'), name_tokens)}"
    )
    return "\n".join(lines)


def parse_result(data: Mapping[str, Any], active_slugs: list[str]) -> LlmResult:
    slug = str(data.get("category", "")).strip()
    try:
        conf = float(data.get("confidence", 0.0))
    except ValueError, TypeError:
        conf = 0.0
    conf = max(0.0, min(1.0, conf)) if isfinite(conf) else 0.0
    reason = str(data.get("reason", ""))[:300]
    if slug != NEEDS_REVIEW and slug not in active_slugs:
        # Preserve the gate that caused review.  The engine uses this reason
        # when creating the durable review decision, so an invalid model slug
        # is not misreported as a deliberate abstention.
        return LlmResult(
            NEEDS_REVIEW,
            conf,
            f"invalid model category slug: {slug}",
        )
    raw_candidates = data.get("candidates", [])
    candidates: list[LlmCandidate] = []
    if isinstance(raw_candidates, Sequence) and not isinstance(
        raw_candidates, (str, bytes)
    ):
        for item in raw_candidates:
            if not isinstance(item, Mapping):
                continue
            candidate = str(item.get("category", item.get("slug", ""))).strip()
            if candidate not in active_slugs or candidate in {
                c.slug for c in candidates
            }:
                continue
            try:
                candidate_conf = float(item.get("confidence", 0.0))
            except ValueError, TypeError:
                candidate_conf = 0.0
            candidates.append(
                LlmCandidate(
                    candidate,
                    max(0.0, min(1.0, candidate_conf))
                    if isfinite(candidate_conf)
                    else 0.0,
                )
            )
            if len(candidates) == 3:
                break
    return LlmResult(slug, conf, reason, tuple(candidates))
