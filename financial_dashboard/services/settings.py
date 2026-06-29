"""DB-backed application settings with in-memory caching.

Settings are stored in a simple key-value ``settings`` table. Metadata
(label, type, category, defaults) lives in SETTINGS_REGISTRY — adding a
new setting only requires a registry entry, no DB migration.

All runtime reads hit an in-memory cache (no DB round-trips in the hot
path). The cache is populated at startup via load_all_settings() and
refreshed on every write.
"""

import json
import logging
from dataclasses import dataclass
from typing import Literal, Mapping

from sqlalchemy import select

from financial_dashboard.config import get_fernet, settings
from financial_dashboard.db import EmailSource, Setting, async_session

logger = logging.getLogger(__name__)


@dataclass
class SettingDef:
    default: str
    data_type: Literal["str", "int", "bool", "json"]
    category: str
    label: str
    description: str = ""
    secret: bool = False
    internal: bool = False  # not shown in the settings UI; never set via the form


SETTINGS_REGISTRY: dict[str, SettingDef] = {
    "telegram.bot_token": SettingDef(
        default="",
        data_type="str",
        category="Telegram",
        label="Bot Token",
        description="Create a bot via @BotFather on Telegram and paste the token here",
        secret=True,
    ),
    "telegram.chat_id": SettingDef(
        default="",
        data_type="int",
        category="Telegram",
        label="Chat ID",
        description="Your Telegram chat ID — send /start to @userinfobot to find it",
    ),
    "telegram.base_url": SettingDef(
        default="",
        data_type="str",
        category="Telegram",
        label="Bot API Base URL",
        description="Override the Bot API server (e.g. a local Bot API server). Leave blank for the default https://api.telegram.org/bot",
    ),
    "telegram.enabled": SettingDef(
        default="false",
        data_type="bool",
        category="Telegram",
        label="Enable Telegram Integration",
    ),
    "telegram.notify_transactions": SettingDef(
        default="true",
        data_type="bool",
        category="Telegram",
        label="Transaction Notifications",
        description="Send a message for each new transaction",
    ),
    "telegram.notify_reminders": SettingDef(
        default="true",
        data_type="bool",
        category="Telegram",
        label="Payment Due Reminders",
        description="Send reminders before credit card due dates",
    ),
    "telegram.notify_payment_received": SettingDef(
        default="true",
        data_type="bool",
        category="Telegram",
        label="Payment Received Detection",
        description="Auto-mark reminders as paid when payment emails arrive",
    ),
    "telegram.bulk_threshold": SettingDef(
        default="5",
        data_type="int",
        category="Telegram",
        label="Bulk Summary After",
        description="Send a summary instead of individual messages above this count",
    ),
    "telegram.reminder_days_before": SettingDef(
        default="[7, 3, 1, 0]",
        data_type="json",
        category="Telegram",
        label="Reminder Schedule",
        description="Days before due date to send reminders",
    ),
    "poll_interval_minutes": SettingDef(
        default="15",
        data_type="int",
        category="Polling",
        label="Poll Interval",
        description="Minutes between email checks",
    ),
    "poll_fetch_limit_per_rule": SettingDef(
        default="50",
        data_type="int",
        category="Polling",
        label="Fetch Limit Per Rule",
        description="Max emails to fetch per rule per cycle",
    ),
    "cas_auto_fetch_enabled": SettingDef(
        default="false",
        data_type="bool",
        category="Net Worth",
        label="Auto-fetch CAS emails",
        description="Poll NSDL-CAS@nsdl.co.in and eCAS@cdslstatement.com daily "
        "and ingest depository CAS PDFs into Net Worth. Requires PAN below.",
    ),
    "cas_pan": SettingDef(
        default="",
        data_type="str",
        category="Net Worth",
        label="CAS PAN",
        description="PAN used to decrypt the CAS PDFs (stored encrypted)",
        secret=True,
    ),
    "gemini.api_key": SettingDef(
        default="",
        data_type="str",
        category="Categorization",
        label="Gemini API Key",
        description="Google Gemini API key used to categorize transactions (stored encrypted)",
        secret=True,
    ),
    "gemini.model": SettingDef(
        default="gemini-2.5-flash",
        data_type="str",
        category="Categorization",
        label="Gemini Model",
        description="Gemini model id used for transaction categorization",
    ),
    "openai.api_key": SettingDef(
        default="",
        data_type="str",
        category="Categorization",
        label="OpenAI API Key",
        description="API key for the OpenAI-compatible categorization provider (stored encrypted)",
        secret=True,
    ),
    "openai.model": SettingDef(
        default="gpt-4o-mini",
        data_type="str",
        category="Categorization",
        label="OpenAI Model",
    ),
    "openai.base_url": SettingDef(
        default="",
        data_type="str",
        category="Categorization",
        label="OpenAI Base URL",
        description="OpenAI-compatible endpoint, e.g. https://host/openai/v1",
    ),
    "categorization.llm_provider": SettingDef(
        default="gemini",
        data_type="str",
        category="Categorization",
        label="LLM Provider",
        description="gemini or openai",
    ),
    "categorization.enabled": SettingDef(
        default="false",
        data_type="bool",
        category="Categorization",
        label="Enable LLM Categorization",
        description="Run the Gemini fallback for transactions the rules can't classify",
    ),
    "categorization.confidence_threshold": SettingDef(
        default="0.6",
        data_type="str",
        category="Categorization",
        label="Confidence Threshold",
        description="LLM results below this confidence route to manual review",
    ),
    "categorization.self_identifiers": SettingDef(
        default="",
        data_type="str",
        category="Categorization",
        label="Self Identifiers",
        description="Comma-separated tokens that identify YOUR OWN accounts — first name, "
        "phone, UPI handles, acct last-4. Drives self_transfer detection AND is censored "
        "before sending to the LLM. e.g. your first name, a phone number, a UPI handle you own",
    ),
    "categorization.hidden_identifiers": SettingDef(
        default="",
        data_type="str",
        category="Categorization",
        label="Hidden Identifiers",
        description="Comma-separated family/private tokens to censor before sending to the "
        "LLM but NOT treat as self. e.g. names of family members whose transfers you receive",
    ),
    "category_vocab_version": SettingDef(
        default="1",
        data_type="int",
        category="Categorization",
        label="Vocabulary Version",
        description="Internal: bumped when a category is added; triggers re-categorization",
        internal=True,
    ),
}

_cache: dict[str, str] = {}


def get_all_settings() -> dict[str, str]:
    """Return a snapshot of all cached settings."""
    return dict(_cache)


def get_setting(key: str, default: str | None = None) -> str | None:
    """Read a setting from cache. Falls back to registry default, then *default*."""
    if key in _cache:
        return _cache[key]
    defn = SETTINGS_REGISTRY.get(key)
    if defn is not None:
        return defn.default
    return default


def get_setting_bool(key: str, default: bool = False) -> bool:
    val = get_setting(key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def get_setting_int(key: str, default: int = 0) -> int:
    val = get_setting(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError, TypeError:
        return default


def get_setting_json(key: str, default=None):
    val = get_setting(key)
    if val is None:
        return default
    try:
        return json.loads(val)
    except json.JSONDecodeError, TypeError:
        return default


def get_self_identifier_tokens() -> tuple[str, ...]:
    """Tokens that identify the account holder's own accounts (for self_transfer detection)."""
    raw = get_setting("categorization.self_identifiers") or ""
    return tuple(t.strip() for t in raw.split(",") if t.strip())


def get_redact_name_tokens() -> tuple[str, ...]:
    """Name fragments to mask before sending text to the LLM.

    Returns the deduplicated union of self_identifiers and hidden_identifiers.
    Both sets are censored before the LLM sees them; only self_identifiers
    also trigger self_transfer detection.
    """
    self_raw = get_setting("categorization.self_identifiers") or ""
    hidden_raw = get_setting("categorization.hidden_identifiers") or ""
    self_tokens = [t.strip() for t in self_raw.split(",") if t.strip()]
    hidden_tokens = [t.strip() for t in hidden_raw.split(",") if t.strip()]
    return tuple(dict.fromkeys(self_tokens + hidden_tokens))


def get_gemini_api_key() -> str:
    """Gemini API key for categorization. DB setting `gemini.api_key` takes
    precedence; falls back to the GEMINI_API_KEY .env value for local runs."""
    return get_setting("gemini.api_key") or settings.gemini_api_key


def get_openai_api_key() -> str:
    """OpenAI-compatible API key. DB setting `openai.api_key` takes precedence;
    falls back to the OPENAI_API_KEY .env value for local runs."""
    return get_setting("openai.api_key") or settings.openai_api_key


def get_openai_base_url() -> str:
    """OpenAI-compatible base URL. DB setting `openai.base_url` takes precedence;
    falls back to the OPENAI_BASE_URL .env value."""
    return get_setting("openai.base_url") or settings.openai_base_url


def get_active_llm_key() -> str:
    """API key for whichever LLM provider is currently selected."""
    provider = get_setting("categorization.llm_provider") or "gemini"
    if provider == "openai":
        return get_openai_api_key()
    return get_gemini_api_key()


def is_telegram_configured() -> bool:
    return (
        get_setting_bool("telegram.enabled")
        and bool(get_setting("telegram.bot_token"))
        and get_setting_int("telegram.chat_id") != 0
    )


def should_notify_transactions() -> bool:
    return is_telegram_configured() and get_setting_bool("telegram.notify_transactions")


def get_telegram_chat_id() -> int:
    return get_setting_int("telegram.chat_id")


def get_telegram_bot_token() -> str:
    return get_setting("telegram.bot_token", "") or ""


def get_telegram_base_url() -> str | None:
    """Optional Bot API base URL override. None falls back to PTB's default
    (https://api.telegram.org/bot).

    A trailing slash is stripped: PTB appends the bot token directly to this
    value, so the URL must end at ``…/bot`` (the default is literally
    ``https://api.telegram.org/bot``). A trailing slash would otherwise yield
    ``…/bot/<token>`` and break every request.
    """
    return (get_setting("telegram.base_url", "") or "").strip().rstrip("/") or None


def get_grouped_settings() -> dict[str, list[dict]]:
    """Build settings grouped by category for the template."""
    current = get_all_settings()
    grouped: dict[str, list[dict]] = {}
    for key, defn in SETTINGS_REGISTRY.items():
        if defn.internal:
            continue
        cat = defn.category
        if cat not in grouped:
            grouped[cat] = []
        val = current.get(key, defn.default)
        grouped[cat].append(
            {
                "key": key,
                "value": val if not defn.secret else "",
                "is_set": bool(val) if defn.secret else None,
                "label": defn.label,
                "type": defn.data_type,
                "description": defn.description,
                "secret": defn.secret,
            }
        )
    return grouped


def parse_form_updates(
    form: Mapping[object, object],
) -> tuple[dict[str, str], list[str]]:
    """Parse and validate a settings form submission.

    Returns (updates, errors). If errors is non-empty, updates should not
    be saved.
    """
    updates: dict[str, str] = {}
    errors: list[str] = []
    for key, defn in SETTINGS_REGISTRY.items():
        if defn.internal:
            continue
        if defn.data_type == "bool":
            updates[key] = "true" if form.get(key) else "false"
        else:
            raw = form.get(key)
            if raw is not None:
                raw = str(raw).strip()
                if defn.secret and raw == "":
                    continue
                if defn.data_type == "int":
                    if not raw:
                        raw = defn.default
                    else:
                        try:
                            int(raw)
                        except ValueError:
                            errors.append(f"{defn.label}: must be a number")
                            continue
                if defn.data_type == "json":
                    if not raw:
                        raw = defn.default
                    else:
                        try:
                            parts = [
                                int(x.strip()) for x in raw.split(",") if x.strip()
                            ]
                            raw = json.dumps(parts)
                        except ValueError, TypeError:
                            errors.append(
                                f"{defn.label}: must be comma-separated numbers"
                            )
                            continue
                updates[key] = raw
    return updates, errors


async def assert_master_key_or_no_secrets(session) -> None:
    """Fail fast at startup if encrypted data exists but no master key is set.

    With EMAIL_SOURCE_MASTER_KEY unset, get_fernet() mints an ephemeral key,
    so any previously-stored secret (email-source credentials, secret settings)
    becomes undecryptable after a restart and load_all_settings() silently
    blanks it. To avoid that data-loss-by-stealth, refuse to boot when such
    data exists. A fresh DB with no secrets boots fine (with a warning).

    Read-only: this only SELECTs, never writes.
    """
    if settings.email_source_master_key:
        return

    has_credentials = (
        await session.execute(
            select(EmailSource.id).where(EmailSource.credentials != "").limit(1)
        )
    ).first() is not None

    secret_keys = [key for key, defn in SETTINGS_REGISTRY.items() if defn.secret]
    has_secret_setting = False
    if secret_keys:
        has_secret_setting = (
            await session.execute(
                select(Setting.key)
                .where(Setting.key.in_(secret_keys))
                .where(Setting.value != "")
                .limit(1)
            )
        ).first() is not None

    if not (has_credentials or has_secret_setting):
        logger.warning(
            "EMAIL_SOURCE_MASTER_KEY is not set. No encrypted data exists yet, "
            "so startup continues — but SET IT before storing any secrets or "
            "they will not survive a restart."
        )
        return

    message = (
        "EMAIL_SOURCE_MASTER_KEY is not set but encrypted data exists in the "
        "database (email-source credentials and/or secret settings). An "
        "ephemeral key would make that data permanently undecryptable. Set "
        "EMAIL_SOURCE_MASTER_KEY in the environment and restart."
    )
    if settings.allow_ephemeral_master_key:
        logger.warning(
            "%s ALLOW_EPHEMERAL_MASTER_KEY is set — continuing anyway; "
            "existing secrets will be unreadable.",
            message,
        )
        return
    raise SystemExit(message)


async def load_all_settings() -> dict[str, str]:
    """Read all rows from DB, merge with registry defaults, populate cache."""
    async with async_session() as session:
        rows = (await session.execute(select(Setting))).scalars().all()

    db_values = {row.key: row.value for row in rows}

    # Decrypt secret fields
    secrets_to_decrypt = {key for key, defn in SETTINGS_REGISTRY.items() if defn.secret}
    for key in secrets_to_decrypt:
        if key in db_values and db_values[key]:
            try:
                db_values[key] = get_fernet().decrypt(db_values[key].encode()).decode()
            except Exception:
                logger.error(
                    "Failed to decrypt setting %s — is EMAIL_SOURCE_MASTER_KEY correct?",
                    key,
                )
                db_values[key] = ""

    merged: dict[str, str] = {}
    for key, defn in SETTINGS_REGISTRY.items():
        merged[key] = db_values.get(key, defn.default)
    for key, val in db_values.items():
        if key not in merged:
            merged[key] = val
    _cache.clear()
    _cache.update(merged)
    return dict(_cache)


async def save_settings(updates: dict[str, str]) -> set[str]:
    """Bulk upsert. Returns the set of keys whose values actually changed."""
    fernet = None
    changed: dict[str, str] = {}

    async with async_session() as session:
        for key, value in updates.items():
            old_value = _cache.get(key)
            if old_value == value:
                continue

            changed[key] = value

            store_value = value
            defn = SETTINGS_REGISTRY.get(key)
            if defn and defn.secret and value:
                if fernet is None:
                    fernet = get_fernet()
                store_value = fernet.encrypt(value.encode()).decode()

            existing = await session.get(Setting, key)
            if existing:
                existing.value = store_value
            else:
                session.add(Setting(key=key, value=store_value))

        if changed:
            await session.commit()
            _cache.update(changed)

    return set(changed)


async def start_services() -> None:
    """Start services based on current settings. Idempotent."""
    if is_telegram_configured():
        # function-local: breaks cycle with services.telegram (telegram imports settings at top)
        from financial_dashboard.services import telegram as telegram_service

        if telegram_service.tg_app is None:
            try:
                await telegram_service.init_telegram(
                    get_telegram_bot_token(), get_telegram_base_url()
                )
            except Exception as e:
                logger.warning("Telegram bot failed to start: %s", e)


async def stop_services() -> None:
    """Stop all managed services."""
    # function-local: breaks cycle with services.telegram
    from financial_dashboard.services import telegram as telegram_service

    await telegram_service.shutdown_telegram()


async def restart_services() -> None:
    """Stop then conditionally restart services based on current settings."""
    await stop_services()
    await start_services()
