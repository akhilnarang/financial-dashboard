"""Path safety for the synthetic loader.

The loader must NEVER target, overwrite, or delete the production database.
This module centralizes every refusal:

* A target database path is resolved to an absolute ``sqlite://`` URL and
  checked against :func:`assert_synthetic_db_path`.
* ``reset`` requires both the explicit ``--reset`` flag and a literal
  confirmation string, and even then only wipes a path that passes the
  synthetic-path check.

The check is conservative: a path must live under a ``synthetic`` directory
and must not be the dashboard's configured production URL. Anything else is
refused with :class:`UnsafeTargetError`.
"""

import os
from pathlib import Path

from scripts.synth.constants import DEFAULT_SYNTHETIC_ROOT


class UnsafeTargetError(ValueError):
    """Raised when a requested DB path could touch non-synthetic data."""


# Path fragments that, if present, mark a directory as the synthetic sandbox.
_SYNTH_MARKERS = ("synthetic",)


def _normalize(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def is_synthetic_path(path: str | Path) -> bool:
    """True iff ``path`` resolves somewhere under a ``synthetic`` directory."""
    resolved = _normalize(path)
    parts = {p.lower() for p in resolved.parts}
    return any(marker in parts for marker in _SYNTH_MARKERS)


def assert_synthetic_db_path(path: str | Path) -> Path:
    """Raise :class:`UnsafeTargetError` unless ``path`` is a synthetic target.

    Refusals:
    * the production dashboard DB (``financial_dashboard.db``),
    * a path that does not contain a ``synthetic`` directory component,
    * a bare ``:memory:`` URL (the loader needs a file so reruns are
      idempotent and verifiable).
    """
    raw = str(path)
    if raw.startswith("sqlite"):
        # strip the sqlite(+aiosqlite)/// prefix for inspection
        stripped = raw.split(":///", 1)[-1]
        if stripped == ":memory:":
            raise UnsafeTargetError(
                "refusing :memory: target; the loader needs a synthetic file DB"
            )
        candidate = Path(stripped)
    else:
        candidate = Path(raw)

    if candidate.name == "financial_dashboard.db":
        raise UnsafeTargetError(
            f"refusing to target the production DB name: {candidate}"
        )

    if not is_synthetic_path(candidate):
        raise UnsafeTargetError(
            f"refusing non-synthetic DB path {candidate}; synthetic output must "
            f"live under a directory named 'synthetic' (default: "
            f"{DEFAULT_SYNTHETIC_ROOT}/)"
        )
    return _normalize(candidate)


def confirm_reset(provided: str | None) -> None:
    """Raise unless ``provided`` is the exact confirmation flag."""
    from scripts.synth.constants import RESET_CONFIRMATION_FLAG

    if provided != RESET_CONFIRMATION_FLAG:
        raise UnsafeTargetError(
            "destructive reset requires both --reset and "
            f"--confirm-reset={RESET_CONFIRMATION_FLAG!r}; refusing to proceed."
        )


def ensure_env_allowed(env_var: str) -> None:
    """Refuse if an environment override points production elsewhere.

    The loader never reads DB_URL from the environment, but if a test/helper
    sets it to a non-synthetic value we bail loudly rather than risk a stray
    production target.
    """
    value = os.environ.get(env_var)
    if not value:
        return
    if value.startswith("sqlite"):
        stripped = value.split(":///", 1)[-1]
        if stripped and stripped != ":memory:":
            try:
                assert_synthetic_db_path(stripped)
            except UnsafeTargetError:
                raise UnsafeTargetError(
                    f"{env_var}={value!r} is not a synthetic target"
                )
