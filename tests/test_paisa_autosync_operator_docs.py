"""Operator-facing pins for the coalesced, transaction-driven Paisa auto-sync.

These tests lock the *contract surfaced to operators* (the contributed setting
defaults + descriptions, the Pydantic DTO defaults, the HTML copy, and the
README/AGENTS documentation) so a future change cannot silently drift the
documented behavior away from the runtime.

The runtime itself (services/paisa/automation.py orchestrator/surface) is
exercised by test_paisa_automation.py and test_paisa_sync_state_schema.py; this
file owns only the manifest/schema/docs surface.
"""

from pathlib import Path

from financial_dashboard.extensions.paisa import PAISA_SETTINGS
from financial_dashboard.schemas.extensions import PaisaConfig, PaisaConfigInput

REPO_ROOT = Path(__file__).resolve().parents[1]
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
AGENTS = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
PAISA_HTML = (
    REPO_ROOT / "financial_dashboard" / "templates" / "extensions" / "paisa.html"
).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Contributed setting: default + description
# --------------------------------------------------------------------------- #


def test_auto_sync_min_interval_default_is_one_minute():
    defn = PAISA_SETTINGS["paisa.auto_sync_min_interval_minutes"]
    assert defn.default == "1"
    assert defn.data_type == "int"


def test_auto_sync_min_interval_description_is_hard_floor_not_debounce():
    desc = PAISA_SETTINGS["paisa.auto_sync_min_interval_minutes"].description
    desc_lower = desc.lower()
    # It is a hard minimum between remote reloads/retries…
    assert "hard minimum" in desc_lower
    assert "reload" in desc_lower
    assert "retr" in desc_lower  # "retry"/"retries"
    # …NOT the event debounce…
    assert "not the event debounce" in desc_lower
    # …and the fixed, non-tunable timings are enumerated.
    assert "5s" in desc
    assert "30s" in desc
    assert "2s" in desc
    assert "1/2/5/10/15" in desc
    # Persisted values are preserved across upgrades.
    assert "preserved" in desc_lower


def test_auto_sync_min_interval_description_dropped_fetch_cycle_debounce_copy():
    """The old copy framed this as a fetch-cycle debounce — that framing is gone."""
    desc = PAISA_SETTINGS["paisa.auto_sync_min_interval_minutes"].description.lower()
    assert "fetch cycles run more often" not in desc
    assert "debounces repeated runs" not in desc


def test_auto_sync_enabled_description_describes_coalescing_not_fetch_cycle():
    desc = PAISA_SETTINGS["paisa.auto_sync_enabled"].description.lower()
    # No longer "after each fetch cycle".
    assert "after each fetch cycle" not in desc
    # Describes the coalesced coordinator + project-mode gate.
    assert "coordinator" in desc or "coalesce" in desc
    assert "project mode" in desc
    # disabled/connect accumulate dirty state with no I/O.
    assert "disabled" in desc and "connect" in desc
    assert "no i/o" in desc or "no io" in desc


# --------------------------------------------------------------------------- #
# Pydantic DTO defaults (the API boundary)
# --------------------------------------------------------------------------- #


def test_paisa_config_dto_default_min_interval_is_one():
    assert PaisaConfig.model_fields["auto_sync_min_interval_minutes"].default == 1


def test_paisa_config_input_dto_default_min_interval_is_one():
    assert PaisaConfigInput.model_fields["auto_sync_min_interval_minutes"].default == 1


# --------------------------------------------------------------------------- #
# HTML template: automation copy + inputs
# --------------------------------------------------------------------------- #


def test_html_auto_sync_checkbox_label_is_coalesced_not_after_fetch():
    # The old label said "Auto sync after fetch (requires project mode)".
    assert "Auto sync after fetch" not in PAISA_HTML
    assert "coalesced" in PAISA_HTML.lower()
    # The project-mode dependency is still communicated.
    assert "requires project mode" in PAISA_HTML.lower()


def test_html_min_interval_label_renamed_and_help_present():
    # Old label was "Auto Sync Min Interval (minutes)"; the input name is stable.
    assert 'name="auto_sync_min_interval_minutes"' in PAISA_HTML
    # Help text surfaces the hard-floor framing + fixed timings.
    assert "hard floor" in PAISA_HTML.lower()
    assert "5s quiet debounce" in PAISA_HTML
    assert "30s max dirty latency" in PAISA_HTML
    assert "2s state polling" in PAISA_HTML
    assert "1/2/5/10/15-min retry backoff" in PAISA_HTML
    assert "six-hour force reload" in PAISA_HTML
    # No per-transaction/partial sync API.
    assert "full-journal" in PAISA_HTML.lower()


def test_html_documents_disabled_connect_accumulate_without_io():
    lower = PAISA_HTML.lower()
    assert "disabled" in lower and "connect" in lower
    assert "no i/o" in lower or "no io" in lower


# --------------------------------------------------------------------------- #
# README documentation pins
# --------------------------------------------------------------------------- #


def test_readme_documents_extension_sync_state_table():
    assert "extension_sync_state" in README
    # Mermaid diagram block present as a standalone table.
    assert "EXTENSION_SYNC_STATE {" in README


def test_readme_model_summary_includes_extension_sync_state():
    assert "| `ExtensionSyncState` |" in README


def test_readme_documents_coalesced_auto_sync_contract():
    # The dedicated Auto sync subsection + transaction-driven coalescing.
    assert "#### Auto sync" in README
    assert "transaction-driven coalescing" in README.lower()
    # Full-journal / no partial API.
    assert "full-journal" in README.lower()
    assert "no per-transaction" in README.lower() or "partial" in README.lower()
    # Exact post-commit semantics.
    assert "post-commit" in README.lower()
    # Fixed timings.
    assert "5s quiet debounce" in README
    assert "30s maximum dirty latency" in README
    assert "every 2s" in README
    assert "1/2/5/10/15" in README
    assert "six-hour force reload" in README
    # Bulk statement behavior.
    assert "bulk statement import" in README.lower()
    assert "one outer commit" in README.lower()
    # disabled/connect accumulate dirty state with no I/O.
    assert "dirty state accumulates" in README.lower()
    assert "no i/o" in README.lower() or "no io" in README.lower()


def test_readme_min_interval_documented_as_hard_floor_default_one():
    # The default changed from 30 to 1 for new installs.
    assert "`paisa.auto_sync_min_interval_minutes` (default **1**)" in README
    assert "hard floor between remote reloads/retries" in README.lower()


def test_readme_key_constraints_cover_sync_state_indexes_and_triggers():
    assert "ix_extension_sync_state_next_attempt" in README
    assert "ix_extension_sync_state_lease_expires" in README
    # CHECK constraints described semantically (desired >= applied, non-negative).
    assert "desired_revision >= applied_revision" in README
    # Trigger tables enumerated.
    for table in (
        "`transactions`",
        "`accounts`",
        "`cards`",
        "`balance_snapshots`",
        "`investment_lots`",
        "`cas_uploads`",
    ):
        assert table in README


def test_readme_request_lifecycle_no_longer_claims_per_fetch_auto_sync():
    # Auto sync is no longer framed as a per-fetch hook in the lifecycle.
    assert (
        "(e.g. an automatic Paisa sync when `paisa.auto_sync_enabled` is on)"
        not in README
    )
    assert "coalesced, transaction-driven coordinator" in README.lower()


# --------------------------------------------------------------------------- #
# AGENTS.md documentation pins
# --------------------------------------------------------------------------- #


def test_agents_documents_coalesced_coordinator_contract():
    assert "coalesced, transaction-driven coordinator" in AGENTS.lower()
    assert "exact post-commit" in AGENTS.lower()
    assert "full-journal" in AGENTS.lower()
    assert "1/2/5/10/15-min retry backoff" in AGENTS
    assert "six-hour force reload" in AGENTS
    assert "auto_sync_min_interval_minutes" in AGENTS
    assert "default **1**" in AGENTS
    # Recursion guard noted.
    assert "recursion guard" in AGENTS.lower()


def test_agents_dedupe_bullet_references_reconcile_not_fetch():
    # The dedupe bullet now talks about a reconcile (not a fetch-cycle run), and
    # a notification failure must not affect reconcile/audit (not "fetch").
    assert "an automatic reconcile fails" in AGENTS.lower()
    assert "reconcile or audit" in AGENTS.lower()
    assert "fetch or audit" not in AGENTS.lower()
