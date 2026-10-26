"""Schemas for merchant-rule endpoints."""

import datetime

from pydantic import BaseModel, ConfigDict, Field


class MerchantRuleRead(BaseModel):
    """One merchant→category rule."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    pattern: str
    category: str
    priority: int
    active: bool
    created_at: datetime.datetime | None


class MerchantRuleListResponse(BaseModel):
    items: list[MerchantRuleRead]


class MerchantRuleSave(BaseModel):
    """Add or update a merchant rule by its normalized pattern."""

    pattern: str = Field(min_length=1, max_length=200)
    category: str = Field(min_length=1, max_length=64)
    priority: int = 100
