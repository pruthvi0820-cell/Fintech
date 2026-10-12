#!/usr/bin/env bash
# FinTray on an always-on Linux server (Ubuntu 22.04/24.04). Run as your normal user (not root):
#   bash deploy/setup_server.sh
# Safe to run again: it only adds what is missing. See docs/CLOUD.md for the whole guide.
set -euo pipefail

REPO_URL="https://github.com/pruthvi0820-cell/Fintech.git"
BRANCH="claude/fintech-ai-trading-assistant-s5z6tf"
APP_DIR="${FINTRAY_DIR:-$HOME/fin_agent_repo}"

if [ "$(id -u)" -eq 0 ]; then
  echo "Run this as your normal user, not root (it asks for sudo when needed)." >&2
  exit 1
fi

echo "== 1/5 System packages"
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip git

echo "== 2/5 FinTray code in $APP_DIR"
if [ ! -d "$APP_DIR/.git" ]; then
  git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
else
  git -C "$APP_DIR" pull --ff-only
fi
cd "$APP_DIR"

echo "== 3/5 Python environment"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[app]"

echo "== 4/5 Settings file (.env)"
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from the example. Edit it: nano $APP_DIR/.env"
fi
chmod 600 .env                     # only you can read your keys

echo "== 5/5 Background services (start on boot, restart if they stop)"
mkdir -p "$HOME/.config/systemd/user"
for unit in fintray-app.service fintray-alerts.service; do
  sed "s#__APP_DIR__#$APP_DIR#g" "deploy/$unit" > "$HOME/.config/systemd/user/$unit"
done
sudo loginctl enable-linger "$USER"  # keep your services running when you log out
systemctl --user daemon-reload
systemctl --user enable --now fintray-app.service fintray-alerts.service

cat <<MSG

Done. FinTray runs at http://127.0.0.1:8501 ON THE SERVER ONLY (not open to the internet).
Next steps (docs/CLOUD.md):
  1. Fill in .env:            nano $APP_DIR/.env    then: systemctl --user restart fintray-app fintray-alerts
  2. Reach it privately from your phone/laptop with Tailscale (step 5 of the guide).
  3. Each trading day, log in to your broker:  cd $APP_DIR && .venv/bin/python scripts/zerodha.py --login
Status: systemctl --user status fintray-app     Logs: journalctl --user -u fintray-app -n 50
MSG
