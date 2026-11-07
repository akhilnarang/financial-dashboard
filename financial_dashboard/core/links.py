"""HTML links to dashboard pages for Telegram messages."""

import html


def page_link(label: str, path: str, base_url: str) -> str:
    """Return the escaped label, linked to its dashboard page when possible.

    Args:
        label: The visible text, for example ``#42``.
        path: The page path, for example ``/transactions/42``.
        base_url: The dashboard base URL. Empty means no link.

    Returns:
        HTML for a Telegram message in HTML parse mode.
    """
    if not base_url:
        return html.escape(label)
    href = html.escape(f"{base_url}{path}", quote=True)
    return f'<a href="{href}">{html.escape(label)}</a>'
