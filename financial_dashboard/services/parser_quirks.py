"""Cross-cutting parser quirks shared by ingestion + matching.

Some bank parsers emit transaction shapes with known ambiguities that
downstream code (the email-side AM/PM disambiguator and the SMS-side
fuzzy-match alias pass) needs to know about. Pulling these constants
into a small module keeps the dependency graph clean: txn_merge.py
needs them but importing from services/emails.py would create a
cross-pipeline import.
"""

# Email types whose body emits transaction time in 12-hour format with
# stripped AM/PM. The email-side disambiguator (services/emails.py
# _disambiguate_am_pm) corrects these at parse time using the email's
# Date header. The fuzzy-match alias pass (services/txn_merge.py
# find_match) uses the same set to know which stored candidates may
# carry a wrong-by-12h transaction_time inherited from pre-fix data.
#
# Add a type here ONLY after confirming with a real sample that the
# source body uses a 12-hour clock with stripped AM/PM. False positives
# corrupt timestamps for non-ambiguous (24-hour) bodies and open the
# alias pass to silent merges of unrelated transactions.
#
# IMPORTANT: the alias-pass merge in services/txn_merge requires
# counterparty agreement on BOTH sides as its primary false-merge
# guard. Before adding an email_type here, confirm the parser always
# emits a non-empty counterparty for this shape — otherwise the alias
# pass refuses to merge and the duplicate-row bug it's trying to fix
# re-emerges. ICICI's icici_cc_transaction_alert parser always sets
# counterparty from the "Info:" field; new entries must offer the
# same guarantee.
#
# Known candidates not yet added (parsers also emit bare HH:MM:SS but
# the clock convention isn't confirmed):
#   - onecard_debit_alert: "Time: 10:30:00" — convention unverified.
AMBIGUOUS_12H_TIME_EMAIL_TYPES: frozenset[str] = frozenset(
    {
        "icici_cc_transaction_alert",
    }
)


# CC payment-received alerts the fuzzy matcher links by card last-4
# instead of counterparty. These carry no merchant and no reference, and
# the two channels split their identity (SMS: time, empty counterparty;
# email: counterparty, no time), so the counterparty gate can't link them
# and they land as duplicate rows. find_match falls back to card last-4
# only for these types and only on a unique candidate. Excludes spend
# alerts, so same-day same-amount swipes never merge on mask alone.
# Add a bank only if it shares this shape: ref-less with an empty
# counterparty on at least one channel (HDFC has a ref, Equitas has a
# counterparty both sides; Axis fits but its email half isn't seen yet).
CARD_PAYMENT_LINK_BY_MASK_EMAIL_TYPES: frozenset[str] = frozenset(
    {
        "icici_cc_payment_received_alert",
        "icici_cc_payment_alert",
    }
)
