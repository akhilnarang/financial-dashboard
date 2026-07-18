"""Paisa extension manifest.

Declares the contributed settings and advertises the setting-contribution,
HTTP/read, and projection capabilities. Upstream Paisa lives at
https://github.com/ananthakumaran/paisa. The HTTP and projection
implementations themselves live outside this module; routes, templates, and
synthetic generation are not part of this manifest.

The JSON-object/list settings (selected_account_ids, account_mappings,
category_mappings, fx_rates) are marked internal because the generic settings
form has no suitable widget for them — they are owned by the Paisa UI.
generated_path is internal too: it is a dashboard-owned operational path, not
user-facing config. ``project_investments`` is a user-facing bool (rendered as a
Paisa UI checkbox); it only takes effect in project mode and never mutates
investment rows.
Marking the JSON settings internal also keeps services.settings.parse_form_updates untouched.
"""

from types import MappingProxyType

from financial_dashboard.extensions.base import (
    Capability,
    ExtensionHealthMeta,
    ExtensionManifest,
    ExtensionNavItem,
)
from financial_dashboard.services.settings import SettingDef

PAISA_CATEGORY = "Paisa"
PAISA_EXTENSION_VERSION = "1.0.0"

PAISA_SETTINGS = MappingProxyType(
    {
        "paisa.mode": SettingDef(
            default="disabled",
            data_type="str",
            category=PAISA_CATEGORY,
            label="Mode",
            description="disabled | connect | project. connect reads from the "
            "Paisa instance; project also projects ledger data into it.",
        ),
        "paisa.base_url": SettingDef(
            default="http://localhost:7500",
            data_type="str",
            category=PAISA_CATEGORY,
            label="Base URL",
            description="Root URL of the local Paisa instance (default port 7500).",
        ),
        "paisa.external_url": SettingDef(
            default="",
            data_type="str",
            category=PAISA_CATEGORY,
            label="External URL",
            description="Public-facing URL of the Paisa instance if different "
            "from the Base URL.",
        ),
        "paisa.allow_remote": SettingDef(
            default="false",
            data_type="bool",
            category=PAISA_CATEGORY,
            label="Allow Remote",
            description="Permit talking to a Paisa instance not on localhost.",
        ),
        "paisa.auth_username": SettingDef(
            default="",
            data_type="str",
            category=PAISA_CATEGORY,
            label="Auth Username",
            description="Paisa X-Auth username.",
        ),
        "paisa.auth_password": SettingDef(
            default="",
            data_type="str",
            category=PAISA_CATEGORY,
            label="Auth Password",
            description="Paisa X-Auth password (stored encrypted).",
            secret=True,
        ),
        "paisa.generated_path": SettingDef(
            default="",
            data_type="str",
            category=PAISA_CATEGORY,
            label="Generated Path",
            description="Exact path of the dashboard-owned generated include "
            "file this dashboard writes (a file, not a directory Paisa writes).",
            internal=True,
        ),
        "paisa.selected_account_ids": SettingDef(
            default="[]",
            data_type="json",
            category=PAISA_CATEGORY,
            label="Selected Account IDs",
            description="Account ids selected for projection. Managed by the Paisa UI.",
            internal=True,
        ),
        "paisa.project_since": SettingDef(
            default="",
            data_type="str",
            category=PAISA_CATEGORY,
            label="Project Since",
            description="Required ISO date cutover for projection; projection "
            "starts from this date.",
        ),
        "paisa.account_mappings": SettingDef(
            default="{}",
            data_type="json",
            category=PAISA_CATEGORY,
            label="Account Mappings",
            description="Map ledger account ids to Paisa accounts. Managed by the Paisa UI.",
            internal=True,
        ),
        "paisa.category_mappings": SettingDef(
            default="{}",
            data_type="json",
            category=PAISA_CATEGORY,
            label="Category Mappings",
            description="Map ledger categories to Paisa accounts. Managed by the Paisa UI.",
            internal=True,
        ),
        "paisa.non_inr_policy": SettingDef(
            default="skip",
            data_type="str",
            category=PAISA_CATEGORY,
            label="Non-INR Policy",
            description="How non-INR transactions are handled. skip drops them; "
            "priced emits them in their own commodity (plus a price directive) "
            "when a paisa.fx_rates rate is configured for the date, and reports "
            "missing_fx_rate otherwise.",
        ),
        "paisa.ledger_cli": SettingDef(
            default="ledger",
            data_type="str",
            category=PAISA_CATEGORY,
            label="Ledger CLI Backend",
            description="Backend the projection targets: ledger | hledger | beancount. "
            "A manual sync requires the upstream Paisa instance to use the same backend.",
        ),
        "paisa.fx_rates": SettingDef(
            default="{}",
            data_type="json",
            category=PAISA_CATEGORY,
            label="FX Rates",
            description="Historical INR-per-unit rates per currency for the priced "
            "non-INR policy. Managed by the Paisa UI as currency/date/rate rows.",
            internal=True,
        ),
        "paisa.report_cache_ttl_seconds": SettingDef(
            default="60",
            data_type="int",
            category=PAISA_CATEGORY,
            label="Report Cache TTL (seconds)",
            description="Server-side TTL for cached curated Paisa report reads. "
            "0 disables caching. A single aggregate refresh coalesces concurrent reads.",
        ),
        "paisa.project_investments": SettingDef(
            default="false",
            data_type="bool",
            category=PAISA_CATEGORY,
            label="Project Investments",
            description="When on and in project mode, additionally emit complete "
            "investment lots (from CAS data) as conservative cost-basis opening "
            "posts. Default off; affects project mode only and never mutates "
            "investment rows.",
        ),
        "paisa.request_timeout_seconds": SettingDef(
            default="15",
            data_type="int",
            category=PAISA_CATEGORY,
            label="Request Timeout (seconds)",
            description="HTTP timeout (seconds) for calls to the Paisa instance.",
        ),
        "paisa.auto_sync_enabled": SettingDef(
            default="false",
            data_type="bool",
            category=PAISA_CATEGORY,
            label="Auto Sync",
            description="When on and in project mode, a coordinator coalesces "
            "dirtying core changes (transactions/accounts/cards/snapshots/CAS "
            "lots/paisa.* settings) into one full-journal Paisa reload. "
            "disabled/connect accumulate dirty state with no I/O. Off by default.",
        ),
        "paisa.auto_sync_min_interval_minutes": SettingDef(
            default="1",
            data_type="int",
            category=PAISA_CATEGORY,
            label="Auto Sync Min Reload Interval (minutes)",
            description="Hard minimum minutes between remote Paisa reloads and "
            "retries — NOT the event debounce. The quiet debounce (5s), max "
            "dirty latency (30s), 2s state polling, lease/single-flight, and "
            "1/2/5/10/15-min retry backoff are fixed and not tunable here. "
            "1 (default) reloads as soon as the coordinator allows; a higher "
            "value throttles a healthy stream and a failing one alike. "
            "Existing persisted values are preserved across upgrades.",
        ),
        "paisa.notify_sync_failures": SettingDef(
            default="false",
            data_type="bool",
            category=PAISA_CATEGORY,
            label="Notify Sync Failures",
            description="Send a Telegram message when an automatic Paisa sync fails. "
            "Requires Telegram to be configured.",
        ),
    }
)

PAISA_EXTENSION = ExtensionManifest(
    id="paisa",
    display_name="Paisa",
    description="Project ledger data into a Paisa instance.",
    extension_version=PAISA_EXTENSION_VERSION,
    capabilities=frozenset(
        {
            Capability.SETTING_CONTRIBUTION,
            Capability.HTTP_READ,
            Capability.PROJECTION,
            Capability.AUTOMATION,
        }
    ),
    navigation=(ExtensionNavItem(label="Paisa", path="/extensions/paisa"),),
    route_prefixes=("/api/extensions/paisa", "/extensions/paisa"),
    health=ExtensionHealthMeta(status_path="/api/extensions/paisa/status"),
    settings=PAISA_SETTINGS,
)
