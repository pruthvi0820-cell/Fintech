# FinTray in the cloud (Step 8)

Why: FinTray and the Telegram alerts keep running with your laptop off, you open it from your phone,
and a cloud server comes with a **static IP**, which brokers require for API orders (since 1 April 2026).

Tags: [Certain] = checked in this project; [Likely] = from provider docs, check the current page;
[Guessing] = a reasonable estimate.

> Prefer code to clicks? [INFRA.md](INFRA.md) builds the same server with Terraform and Ansible, plus nightly
> S3 backups.

## The one safety rule

**FinTray has no login screen.** Anyone who can open the page sees your holdings and journal, and,
if you enabled real orders, the order buttons. So it is set up to listen **only on the server
itself** (127.0.0.1), and you reach it through **Tailscale**, a free private network that only your
own devices can join. Never open port 8501 to the internet, and never use `tailscale funnel` (that
one IS public). [Certain: the setup script and services listen on 127.0.0.1 only]

## 1. Get a server (about 15 minutes)

What you need: **Ubuntu 22.04 or 24.04**, at least **2 CPUs and 2 GB RAM** (4 GB is comfortable),
in an **Indian region** (Mumbai/Hyderabad) for speed. [Likely]

| Option | Cost | Note |
|---|---|---|
| Oracle Cloud "Always Free" (Mumbai/Hyderabad) | ₹0 | Free ARM servers; sign-up needs a card; free capacity is sometimes "out of stock" [Likely] |
| AWS Lightsail (Mumbai) | about $5–12 a month | Simplest screens; static IP is free while attached [Likely] |
| DigitalOcean (Bangalore) | about $6–12 a month | "Reserved IP" for the static address [Likely] |

When creating it:
- choose **SSH key** login (the provider's page shows how), not a password;
- **reserve a static (public) IP** and attach it to the server. Write it down: it goes to your
  broker later.

## 2. Log in to the server

From your laptop's cmd (the provider shows the exact command and user name):
```
ssh ubuntu@YOUR.STATIC.IP
```

## 3. Firewall: allow only SSH [Likely]

```
sudo ufw allow OpenSSH
sudo ufw enable
```
(Tailscale works through this; nothing else needs to be open.)

## 4. Install FinTray (one command)

```
git clone --branch claude/fintech-ai-trading-assistant-s5z6tf https://github.com/pruthvi0820-cell/Fintech.git ~/fin_agent_repo
bash ~/fin_agent_repo/deploy/setup_server.sh
```
It installs Python, FinTray and two background services that start on boot and restart if they stop:
`fintray-app` (the page) and `fintray-alerts` (Telegram). Then fill in your settings:
```
nano ~/fin_agent_repo/.env
systemctl --user restart fintray-app fintray-alerts
```
Copy the same lines you have in your laptop's `.env` (Telegram token and chat id, broker key and
secret). Type them; don't send them through chat apps.

## 5. Open it from your phone: Tailscale (free) [Likely: check `tailscale serve --help`]

On the server:
```
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
sudo tailscale serve --bg 8501
```
`tailscale up` prints a link: open it and log in (Google account is fine). On your **phone and
laptop**, install the Tailscale app and log in with the **same** account. Then open the address
`tailscale serve status` shows (like `https://your-server.tailnet-name.ts.net`). Only devices in your
Tailscale account can open it.

## 6. Bring your data (optional)

Your journal, paper trades, notes and library are files on your laptop. To move them, from the
laptop's cmd in `D:\projects\fin_agent_repo`:
```
scp -r journal notes library ubuntu@YOUR.STATIC.IP:~/fin_agent_repo/
```
Then `systemctl --user restart fintray-app` on the server.

## 7. The AI on the server

The local AI model (Ollama) needs a GPU or a lot of memory to be fast; on a small cloud server it is
very slow. [Likely] Options: everything except "Explain with AI" and the chat works without it; or
set `FIN_AGENT_PROVIDER=openai_compat` with a free-tier API in `.env` (see `.env.example`); or keep
using the AI on your laptop.

## 8. Every trading day

1. Log in to your broker (the daily login):
   ```
   ssh ubuntu@YOUR.STATIC.IP
   cd ~/fin_agent_repo && .venv/bin/python scripts/zerodha.py --login
   ```
   (Set your broker app's redirect URL to `http://127.0.0.1:8501/`; after logging in, copy the
   address from your browser into the SSH window, as on the laptop.)
2. Open FinTray on your phone through Tailscale.

## 9. Real orders from the server (Step 7)

1. Register the server's **static IP** in your broker's developer app (Zerodha: developers.kite.trade).
2. Add `FIN_AGENT_LIVE_TRADING=true` to `.env` and restart: `systemctl --user restart fintray-app`.
3. Every order still needs your typed approval in FinTray and passes the risk rules.

## 10. Updating, checking, stopping

```
cd ~/fin_agent_repo && git pull && .venv/bin/python -m pip install -e ".[app]"
systemctl --user restart fintray-app fintray-alerts
systemctl --user status fintray-app          # is it running?
journalctl --user -u fintray-alerts -n 50    # last 50 lines of the alerts log
systemctl --user stop fintray-app fintray-alerts
```
Paid servers cost money while they exist, even when stopped (the provider's billing page shows
it). [Likely] Delete the server and release the static IP when you no longer need them.
