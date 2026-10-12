"""Angel One read-only connection: daily login and a quick look at your account.

Usage:
    python scripts/angelone.py --login      # once a day: type your PIN (hidden) and the 6-digit app code
    python scripts/angelone.py --check      # who is logged in, holdings count, today's trades count
    python scripts/angelone.py --holdings   # your holdings
    python scripts/angelone.py --logout     # forget today's login on this computer

Needs FIN_AGENT_ANGEL_API_KEY and FIN_AGENT_ANGEL_CLIENT_CODE in .env. Your PIN and code are typed
here, sent to Angel One once, and never stored or shown. Nothing here can place an order.
"""

from __future__ import annotations

import argparse
import getpass
import sys

from fin_agent import config  # noqa: F401  (loads .env)
from fin_agent.broker import angelone


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Angel One read-only connection.")
    g = p.add_mutually_exclusive_group(required=True)
    for flag in ("--login", "--check", "--holdings", "--logout"):
        g.add_argument(flag, action="store_true")
    args = p.parse_args()
    try:
        if args.logout:
            angelone.logout()
            print("Logged out on this computer.")
            return 0
        if args.login:
            cfg = angelone.AngelConfig.from_env()
            print(f"Angel One login for client {cfg.client_code}. Type on this computer only; nothing is stored.")
            pin = getpass.getpass("PIN (hidden while typing): ")
            totp = input("6-digit code from your authenticator app: ")
            angelone.save_session(angelone.login(cfg, pin, totp))
            print("✅ Logged in to Angel One for today (read-only in FinTray).")
            return 0
        api = angelone.connect()
        if args.holdings:
            port = angelone.holdings_to_portfolio(api.holdings())
            if not port.holdings:
                print("No holdings in this Angel One account.")
            for h in port.holdings:
                avg = f"avg ₹{h.avg_cost:,.2f}" if h.avg_cost else "avg ?"
                last = f"last ₹{h.last_price:,.2f}" if h.last_price else "last ?"
                print(f"{h.symbol:15} {h.quantity:>8g} shares  {avg:>16}  {last:>16}")
        else:
            prof = api.profile()
            print(f"Logged in: {prof.get('name', '?')} ({prof.get('clientcode', '?')})")
            print(f"Holdings: {len(api.holdings())}")
            print(f"Trades today: {len(api.trades())}")
            print("✅ Angel One connection works.")
        return 0
    except angelone.AngelError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
