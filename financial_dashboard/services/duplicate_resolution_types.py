from dataclasses import dataclass
from typing import NamedTuple

from financial_dashboard.db import Email, FetchRule, Transaction
from financial_dashboard.schemas.emails import TransactionEnrichmentState
from financial_dashboard.services.txn_merge import EnrichmentDiff


class EligibleResolutionRows(NamedTuple):
    email: Email
    target: Transaction
    rule: FetchRule


@dataclass(frozen=True)
class ResolutionEvaluation:
    email: Email
    target: Transaction
    txn_data: dict
    diff: EnrichmentDiff
    token: str
    before: TransactionEnrichmentState
    after: TransactionEnrichmentState
