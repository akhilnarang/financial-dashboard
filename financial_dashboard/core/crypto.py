"""Fernet encryption helpers for email source credentials."""

import json

from financial_dashboard.config import get_fernet


def encrypt_credentials(creds: dict) -> str:
    """Encrypt a credentials dict to a Fernet token string."""
    f = get_fernet()
    return f.encrypt(json.dumps(creds).encode()).decode()


def decrypt_credentials(token: str) -> dict:
    """Decrypt a Fernet token string back to a credentials dict."""
    f = get_fernet()
    return json.loads(f.decrypt(token.encode()))
