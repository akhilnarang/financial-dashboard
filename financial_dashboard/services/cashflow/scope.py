"""Account scope: which accounts a cashflow figure is allowed to count.

A cash-basis report is a report about one perimeter — money moving through the
bank. So every figure, every footnote and every drill-through link has to agree
on what "the bank" means, and they can only agree if the rule is written once.
That rule is a SQL predicate over ``transactions``, not a Python test on a
loaded row: the report aggregates in a ``GROUP BY`` and never sees rows, so a
per-row account lookup could not scope it at all.

The three scopes partition the table — every transaction is in exactly one:

* ``bank``: linked to a ``bank_account`` or a ``debit_card``. A debit card is
  immediate bank cash movement, so it belongs on the cash-basis side even
  though nothing is currently typed that way.
* ``card``: linked to a ``credit_card``. Card swipes are not bank movements;
  the bank pays for them later, as one bill.
* ``unaccounted``: linked to nothing (``account_id IS NULL``), linked to an
  account row that is gone, or linked to an account whose ``type`` is a string
  none of the above name. ``accounts.type`` has no CHECK constraint, so an
  unknown type is always possible and must land somewhere rather than
  disappearing between two predicates.

Written as ``EXISTS`` rather than an outer join with a NULL test, because
``EXISTS`` is two-valued: ``~EXISTS`` is true for the unlinked row, where
``NOT (account_id IN (...))`` would be NULL and drop it. The unaccounted scope
is the complement of the two known ones by construction, so the partition
cannot develop a hole when a new account type is added to one list and not the
other.
"""

from typing import Literal

from sqlalchemy import ColumnElement, exists, select

from financial_dashboard.db.models import Account, Transaction
from financial_dashboard.services.categorization.slugs import CREDIT_CARD_ACCOUNT_TYPE

#: The account scopes a figure (or a ``/transactions`` listing) can be cut to.
#: Absent scope always means "every account", which is a fourth thing and
#: deliberately not a member here.
Scope = Literal["bank", "card", "unaccounted"]

BANK_ACCOUNT_TYPES = ("bank_account", "debit_card")
CARD_ACCOUNT_TYPES = (CREDIT_CARD_ACCOUNT_TYPE,)
KNOWN_ACCOUNT_TYPES = BANK_ACCOUNT_TYPES + CARD_ACCOUNT_TYPES


def _linked_to(*types: str) -> ColumnElement[bool]:
    # Correlated on purpose: the subquery reads `transactions` from whichever
    # aggregate encloses it, so scoping a query costs no extra statement.
    return exists(
        select(Account.id)
        .where(Account.id == Transaction.account_id, Account.type.in_(types))
        .correlate(Transaction)
    )


BANK_SCOPE = _linked_to(*BANK_ACCOUNT_TYPES)
CARD_SCOPE = _linked_to(*CARD_ACCOUNT_TYPES)
UNACCOUNTED_SCOPE = ~_linked_to(*KNOWN_ACCOUNT_TYPES)

SCOPE_PREDICATES: dict[Scope, ColumnElement[bool]] = {
    "bank": BANK_SCOPE,
    "card": CARD_SCOPE,
    "unaccounted": UNACCOUNTED_SCOPE,
}


def scope_predicate(scope: Scope | None) -> ColumnElement[bool] | None:
    """The SQL clause that cuts ``transactions`` down to one account scope.

    ``None`` — no scope asked for — returns ``None`` rather than a
    true-for-everything clause, so a caller can tell "every account" apart from
    "one of the three scopes" and leave its query untouched.
    """
    return None if scope is None else SCOPE_PREDICATES[scope]
