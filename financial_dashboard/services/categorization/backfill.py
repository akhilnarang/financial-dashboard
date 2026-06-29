"""Resumable backfill: run the rule sweep to completion (seeds few-shot
examples), then loop the LLM sweep until it stops making progress."""

import logging
from typing import NamedTuple

from financial_dashboard.services.categorization.sweep import (
    run_llm_sweep,
    run_rule_sweep,
)

logger = logging.getLogger(__name__)


class BackfillResult(NamedTuple):
    rules_processed: int
    llm_categorized: int


async def run_backfill(
    *, rules_only: bool = False, batch_size: int = 100
) -> BackfillResult:
    # run_rule_sweep returns rows PROCESSED (each → 'rule' or 'pending_llm'), so
    # the never-touched set strictly shrinks and this terminates with full coverage.
    total_rules = 0
    while True:
        n = await run_rule_sweep(batch_limit=max(batch_size, 500))
        total_rules += n
        if n == 0:
            break
    logger.info("Rule pass processed %d rows", total_rules)

    if rules_only:
        return BackfillResult(total_rules, 0)

    total_llm = 0
    while True:
        n = await run_llm_sweep(batch_limit=batch_size)
        total_llm += n
        if n == 0:
            break
    logger.info("LLM sweep categorized %d rows", total_llm)
    return BackfillResult(total_rules, total_llm)
