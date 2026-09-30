"""Telegram-safe rendering primitives shared by assistant delivery paths."""

TELEGRAM_MESSAGE_LIMIT = 4096


def split_plain_text(text: str, *, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Split plain text at useful boundaries."""
    remaining = text.strip()
    chunks: list[str] = []
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit + 1)
        if split_at < limit // 2:
            split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at < 1:
            split_at = limit
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    chunks.append(remaining)
    return chunks
