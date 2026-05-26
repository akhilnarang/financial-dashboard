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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Mapping

from sqlalchemy import select

from financial_dashboard.config import get_fernet
from financial_dashboard.db import Setting, async_session

logger = logging.getLogger(__name__)


@dataclass
class SettingDef:
    default: str
    data_type: Literal["str", "int", "bool", "json"]
    category: str
    label: str
    description: str = ""
    secret: bool = False


@dataclass
class PaisaConfig:
    ledger_cli: str
    main_journal_path: str
    generated_journal_path: str
    default_expense_account: str
    default_income_account: str
    fallback_asset_account: str
    fallback_liability_account: str
    account_map: dict[str, str]


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
    "ledger.backend": SettingDef(
        default="local",
        data_type="str",
        category="Ledger",
        label="Ledger Backend",
        description="Use local dashboard ledger or paisa export bridge mode",
    ),
    "paisa.ledger_cli": SettingDef(
        default="ledger",
        data_type="str",
        category="Ledger",
        label="Paisa Ledger CLI",
        description="v1 supports only `ledger`",
    ),
    "paisa.main_journal_path": SettingDef(
        default="",
        data_type="str",
        category="Ledger",
        label="Paisa Main Journal Path",
        description="Absolute path to your main Paisa journal file",
    ),
    "paisa.generated_journal_path": SettingDef(
        default="",
        data_type="str",
        category="Ledger",
        label="Paisa Generated Journal Path",
        description="Output include file path under the main journal directory",
    ),
    "paisa.default_expense_account": SettingDef(
        default="Expenses:Uncategorized",
        data_type="str",
        category="Ledger",
        label="Default Expense Account",
    ),
    "paisa.default_income_account": SettingDef(
        default="Income:Uncategorized",
        data_type="str",
        category="Ledger",
        label="Default Income Account",
    ),
    "paisa.fallback_asset_account": SettingDef(
        default="Assets:Unknown",
        data_type="str",
        category="Ledger",
        label="Fallback Asset Account",
    ),
    "paisa.fallback_liability_account": SettingDef(
        default="Liabilities:Unknown",
        data_type="str",
        category="Ledger",
        label="Fallback Liability Account",
    ),
    "paisa.account_map": SettingDef(
        default="{}",
        data_type="json",
        category="Ledger",
        label="Paisa Account Map",
        description="Raw JSON object mapping source ids to ledger accounts",
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


def get_ledger_backend() -> Literal["local", "paisa"]:
    value = (get_setting("ledger.backend", "local") or "local").strip().lower()
    if value == "paisa":
        return "paisa"
    return "local"


def _get_paisa_config_values(
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    values: dict[str, str] = {}
    for key in SETTINGS_REGISTRY:
        val = get_setting(key, SETTINGS_REGISTRY[key].default)
        values[key] = val if val is not None else SETTINGS_REGISTRY[key].default
    if overrides:
        for key, value in overrides.items():
            values[key] = value
    return values


def _parse_paisa_account_map(raw: str, *, strict: bool) -> dict[str, str]:
    payload = raw.strip() if raw else "{}"
    if not payload:
        payload = "{}"
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        if strict:
            raise ValueError("paisa.account_map must be valid JSON") from exc
        return {}

    if not isinstance(parsed, dict):
        if strict:
            raise ValueError("paisa.account_map must be a JSON object")
        return {}

    normalized: dict[str, str] = {}
    for key, value in parsed.items():
        if strict and (not isinstance(key, str) or not isinstance(value, str)):
            raise ValueError("paisa.account_map keys and values must be strings")
        normalized[str(key)] = str(value)
    return normalized


def _is_safe_ledger_account_name(value: str) -> bool:
    if not value.strip():
        return False
    forbidden = (";", "\n", "\r", "\t")
    return not any(token in value for token in forbidden)


def _build_paisa_config(
    values: Mapping[str, str], *, strict_account_map: bool
) -> PaisaConfig:
    return PaisaConfig(
        ledger_cli=(values.get("paisa.ledger_cli") or "").strip(),
        main_journal_path=(values.get("paisa.main_journal_path") or "").strip(),
        generated_journal_path=(
            values.get("paisa.generated_journal_path") or ""
        ).strip(),
        default_expense_account=(
            values.get("paisa.default_expense_account")
            or SETTINGS_REGISTRY["paisa.default_expense_account"].default
        ).strip(),
        default_income_account=(
            values.get("paisa.default_income_account")
            or SETTINGS_REGISTRY["paisa.default_income_account"].default
        ).strip(),
        fallback_asset_account=(
            values.get("paisa.fallback_asset_account")
            or SETTINGS_REGISTRY["paisa.fallback_asset_account"].default
        ).strip(),
        fallback_liability_account=(
            values.get("paisa.fallback_liability_account")
            or SETTINGS_REGISTRY["paisa.fallback_liability_account"].default
        ).strip(),
        account_map=_parse_paisa_account_map(
            values.get("paisa.account_map", "{}"),
            strict=strict_account_map,
        ),
    )


def validate_paisa_config(
    values: Mapping[str, str] | None = None,
) -> PaisaConfig:
    snapshot = _get_paisa_config_values(values)
    backend = (snapshot.get("ledger.backend") or "local").strip().lower()
    backend_value: Literal["local", "paisa"] = (
        "paisa" if backend == "paisa" else "local"
    )
    config = _build_paisa_config(
        snapshot,
        strict_account_map=(backend_value == "paisa"),
    )

    account_fields = {
        "paisa.default_expense_account": config.default_expense_account,
        "paisa.default_income_account": config.default_income_account,
        "paisa.fallback_asset_account": config.fallback_asset_account,
        "paisa.fallback_liability_account": config.fallback_liability_account,
    }
    for field_key, account_name in account_fields.items():
        if not _is_safe_ledger_account_name(account_name):
            raise ValueError(
                f"{field_key} must be a valid ledger account name "
                "(no semicolons, newlines, carriage returns, or tabs)"
            )

    if backend_value != "paisa":
        return config

    for map_key, account_name in config.account_map.items():
        if not _is_safe_ledger_account_name(account_name):
            raise ValueError(
                "paisa.account_map values must be valid ledger account names "
                f"(invalid key: {map_key})"
            )

    if config.ledger_cli != "ledger":
        raise ValueError("paisa.ledger_cli must be 'ledger' when ledger.backend=paisa")
    if not config.main_journal_path:
        raise ValueError(
            "paisa.main_journal_path is required when ledger.backend=paisa"
        )
    if not config.generated_journal_path:
        raise ValueError(
            "paisa.generated_journal_path is required when ledger.backend=paisa"
        )

    main_path_raw = Path(config.main_journal_path).expanduser()
    if not main_path_raw.is_absolute():
        raise ValueError("paisa.main_journal_path must be an absolute path")

    generated_path_raw = Path(config.generated_journal_path).expanduser()
    if not generated_path_raw.is_absolute():
        raise ValueError("paisa.generated_journal_path must be an absolute path")

    main_path = main_path_raw.resolve(strict=False)
    generated_path = generated_path_raw.resolve(strict=False)
    if generated_path == main_path:
        raise ValueError(
            "paisa.generated_journal_path must not be the main journal file"
        )

    main_dir = main_path.parent
    if not generated_path.is_relative_to(main_dir):
        raise ValueError(
            "paisa.generated_journal_path must be in the main journal directory or a subdirectory"
        )

    return replace(
        config,
        main_journal_path=str(main_path),
        generated_journal_path=str(generated_path),
    )


def get_paisa_config() -> PaisaConfig:
    return validate_paisa_config()


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


def get_grouped_settings() -> dict[str, list[dict]]:
    """Build settings grouped by category for the template."""
    current = get_all_settings()
    grouped: dict[str, list[dict]] = {}
    for key, defn in SETTINGS_REGISTRY.items():
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
                        if key == "telegram.reminder_days_before":
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
                        else:
                            try:
                                parsed = json.loads(raw)
                            except json.JSONDecodeError:
                                errors.append(f"{key}: must be valid JSON")
                                continue
                            if key == "paisa.account_map" and not isinstance(
                                parsed, dict
                            ):
                                errors.append(
                                    "paisa.account_map: must be a JSON object"
                                )
                                continue
                            raw = json.dumps(parsed)
                updates[key] = raw
    return updates, errors


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
    validate_paisa_config(merged)
    _cache.clear()
    _cache.update(merged)
    return dict(_cache)


async def save_settings(updates: dict[str, str]) -> set[str]:
    """Bulk upsert. Returns the set of keys whose values actually changed."""
    validate_paisa_config(_get_paisa_config_values(updates))

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
    validate_paisa_config()

    if is_telegram_configured():
        # function-local: breaks cycle with services.telegram (telegram imports settings at top)
        from financial_dashboard.services import telegram as telegram_service

        if telegram_service.tg_app is None:
            try:
                await telegram_service.init_telegram(get_telegram_bot_token())
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
