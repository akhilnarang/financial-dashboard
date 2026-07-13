"""Category slugs that more than one module has to name literally.

A slug spelled as a bare string in two places is two places to change, and the
second one is always the one that gets missed. Anything here is a slug whose
*identity* is load-bearing somewhere outside the vocabulary itself — a default,
a sentinel, or a reporting special case — so it gets one name that every user
imports rather than a repeated literal.

Deliberately import-free: every layer (categorization, reporting, the web
routes) can depend on it without pulling a session, a setting or a model in.
"""

# The category a credit falls back to when nothing better fits, and the one the
# cashflow report breaks out on its own line rather than counting as income:
# somebody handing money back is not money earned.
REPAYMENT_SLUG = "repayment"

# Money moving onto a credit card to settle its bill — internal churn, so the
# cashflow report counts it as neither income nor spend:
CREDIT_CARD_PAYMENT_SLUG = "credit_card_payment"

# A merchant handing money back:
REFUND_SLUG = "refund"

# Not a category — the sentinel a classifier writes when it could not pick one.
UNKNOWN_SLUG = "unknown"

# accounts.type of a credit-card account. Load-bearing outside the vocabulary:
# a card can only ever be credited by a refund or a bill payment, so the
# categorizer has to branch on it.
CREDIT_CARD_ACCOUNT_TYPE = "credit_card"
