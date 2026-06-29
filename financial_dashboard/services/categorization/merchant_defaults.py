"""Built-in default merchant→category rules, seeded by init_db.

These are GENERIC, widely-encountered brands/services (national merchants,
common payroll/cards/investment rails) — useful to any user out of the box and
carrying no personal data, so they ship in version control. They seed the
DB-backed merchant_rules table via INSERT OR IGNORE, so a user can edit or
delete any of them at runtime (scripts/merchant_rules.py), and personal/local
overrides come from the untracked merchant_seed_data.py on top.

{category: [patterns]} — each pattern is a lowercased substring matched against
normalize_text(counterparty + ' ' + raw_description).
"""

DEFAULT_MERCHANT_RULES: dict[str, list[str]] = {
    "bill_payment": ["bajaj finance", "bajajfinance", "bajajfinserv", "amica"],
    "credit_card_payment": [
        "credit card bill payment",
        "innopay",
        "cred club",
        "payment on cred",
        "bbps payment received",
        "payment received",
        "repayment thank you",
        "cheq digit",
        "vi ind rt",
        "paid via navi",
        "navitechnologie",
        "navircbp",
        "cf navitechnolo",
    ],
    "salary": ["rippling", "people center inc", "nium pte", "payroll"],
    "investment": [
        "zerodha",
        "iccl",
        "raise securities",
        "raise se",
        "raisesecurities",
        "employee provident fund",
        "epfo",
    ],
    "dining": [
        "swiggy",
        "zomato",
        "eazydiner",
        "gustoso",
        "the coffee machine",
        "mcdonald",
        "kfc",
        "california burrito",
        "coffee nation",
        "two good sisters",
    ],
    "groceries": ["blinkit", "zepto", "instamart", "bigbasket", "jp2100001"],
    "shopping": [
        "ishop",
        "asspl",
        "amazon",
        "flipkart",
        "myntra",
        "ajio",
        "nykaa",
        "croma",
        "uniqlo",
        "reward 360",
        "reward360",
        "cred voucher",
    ],
    "transport": ["uber", "pune metro", "careem", "taxi", "limousine"],
    "car_maintenance": ["skoda", "wonder cars"],
    "healthcare": ["noble plus", "reliance foundat", "apollo"],
    "fuel": ["bharat petroleu"],
    "travel": ["makemytrip", "irctc", "loungeone", "pay www lou", "pax innovat"],
    "entertainment": ["bookmyshow", "orbgen"],
    "cashback_rewards": [
        "supermoney",
        "converted to statement credit",
        "cashback",
        "poweraccess pa",
        "poweraccess on",
    ],
    "refund": ["poweraccess cr"],
    "utilities": ["hathway", "airtel"],
    "self_transfer": ["addmoney", "walletwithd"],
    "gift": ["shaadi", "shagun", "birthday", "anniversary", "belated"],
}
