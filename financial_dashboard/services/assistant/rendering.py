"""Telegram-safe rendering primitives shared by assistant delivery paths."""

TELEGRAM_MESSAGE_LIMIT = 4096


def split_plain_text(
    text: str,
    *,
    footer: str = "",
    limit: int = TELEGRAM_MESSAGE_LIMIT,
) -> list[str]:
    """Split plain text at useful boundaries and reserve space for a footer."""
    clean = text.strip()
    suffix = f"\n\n{footer.strip()}" if footer.strip() else ""
    capacity = limit - len(suffix)
    if capacity < 1:
        raise ValueError("Telegram footer leaves no room for message text")
    if not clean:
        return [suffix.lstrip()]

    chunks: list[str] = []
    remaining = clean
    while len(remaining) > capacity:
        split_at = remaining.rfind("\n", 0, capacity + 1)
        if split_at < capacity // 2:
            split_at = remaining.rfind(" ", 0, capacity + 1)
        if split_at < 1:
            split_at = capacity
        chunks.append(remaining[:split_at].rstrip() + suffix)
        remaining = remaining[split_at:].lstrip()
    chunks.append(remaining + suffix)
    return chunks
