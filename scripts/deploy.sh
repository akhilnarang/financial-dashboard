#!/bin/sh
# Deploy: pull the latest, sync the venv, install the unit, and restart the
# service. The unit is installed every time, so a unit change ships with the
# deploy.
#
# `uv sync` runs here (in the user's environment, which has git auth) rather
# than in the unit, because the systemd environment cannot fetch git deps.
set -eu

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"

uv=$(command -v uv 2>/dev/null || echo "$HOME/.local/bin/uv")
unit_name=financial-dashboard.service
unit_dir=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user

git pull --ff-only
"$uv" sync --locked --no-dev
install -d -m 0755 "$unit_dir"
install -m 0644 "$root/deploy/systemd/$unit_name" "$unit_dir/$unit_name"
systemctl --user daemon-reload
systemctl --user restart "$unit_name"
systemctl --user --no-pager status "$unit_name" | head -n 8
