#!/bin/sh
# Install the financial-dashboard systemd user unit (one-time per host).
#
# Copies deploy/systemd/financial-dashboard.service into the user unit dir,
# syncs the venv without dev dependencies (the unit runs --no-sync), enables
# the unit, and restarts it. Re-run to pick up unit changes. Stop any manual
# `uv run uvicorn` first so the unit can bind port 8000.
set -eu

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
unit_name=financial-dashboard.service
unit_src=$root/deploy/systemd/$unit_name
unit_dir=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user

uv=$(command -v uv 2>/dev/null || echo "$HOME/.local/bin/uv")
[ -x "$uv" ] || { echo "uv not found at $uv" >&2; exit 1; }

install -d -m 0755 "$unit_dir"
install -m 0644 "$unit_src" "$unit_dir/$unit_name"

cd "$root"
"$uv" sync --locked --no-dev

systemctl --user daemon-reload
systemctl --user enable "$unit_name"
# A restart also starts a stopped unit. "enable --now" does not restart a
# running unit, so it would keep the old ExecStart.
systemctl --user restart "$unit_name"
systemctl --user --no-pager status "$unit_name" | head -n 8
