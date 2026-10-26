"""Merchant-rule endpoints.

Every write commits and then reloads the in-process rule cache, so the next
categorization uses the new rules without a restart.
"""

from typing import Annotated

from fastapi import APIRouter, Path, Response, status

from financial_dashboard.core.deps import AsyncSessionDep
from financial_dashboard.exceptions import (
    NotFoundException,
    UnprocessableEntityException,
)
from financial_dashboard.schemas.common import DatabaseId
from financial_dashboard.schemas.merchant_rules import (
    MerchantRuleListResponse,
    MerchantRuleRead,
    MerchantRuleSave,
)
from financial_dashboard.services.categorization.merchant_rules import (
    deactivate_merchant_rule,
    list_merchant_rules,
    load_merchant_rules,
    save_merchant_rule,
)

router = APIRouter()


@router.get("/merchant-rules")
async def merchant_rules_list(session: AsyncSessionDep) -> MerchantRuleListResponse:
    """List every merchant rule in match order.

    Args:
        session: Request session.

    Returns:
        All rules, active and inactive.
    """
    rules = await list_merchant_rules(session)
    return MerchantRuleListResponse.model_validate({"items": rules})


@router.post("/merchant-rules")
async def merchant_rule_save(
    payload: MerchantRuleSave, session: AsyncSessionDep
) -> MerchantRuleRead:
    """Add a merchant rule, or update the rule with the same pattern.

    Args:
        payload: Pattern, category slug and priority.
        session: Request session.

    Returns:
        The stored rule with its normalized pattern.
    """
    try:
        rule = await save_merchant_rule(
            session, payload.pattern, payload.category, priority=payload.priority
        )
    except ValueError as exc:
        raise UnprocessableEntityException(detail=str(exc)) from exc
    response = MerchantRuleRead.model_validate(rule)
    await session.commit()
    await load_merchant_rules(session)
    return response


@router.delete("/merchant-rules/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def merchant_rule_delete(
    rule_id: Annotated[DatabaseId, Path()], session: AsyncSessionDep
) -> Response:
    """Deactivate one merchant rule so categorization stops using it.

    Args:
        rule_id: Primary key of the rule.
        session: Request session.

    Returns:
        An empty 204 response.
    """
    if not await deactivate_merchant_rule(session, rule_id):
        raise NotFoundException(detail="Active merchant rule not found")
    await session.commit()
    await load_merchant_rules(session)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
