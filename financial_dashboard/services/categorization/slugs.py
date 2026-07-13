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

# Not a category — the sentinel a classifier writes when it could not pick one.
UNKNOWN_SLUG = "unknown"
