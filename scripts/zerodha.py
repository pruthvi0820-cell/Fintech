"""Zerodha read-only connection: daily login and a quick look at your account.

Usage:
    python scripts/zerodha.py --login       # once a day: log in on Zerodha's site, paste the address back
    python scripts/zerodha.py --holdings    # your holdings, from Zerodha
    python scripts/zerodha.py --check       # who is logged in, funds, open positions, today's orders
    python scripts/zerodha.py --logout      # forget today's login on this computer

Needs FIN_AGENT_KITE_API_KEY and FIN_AGENT_KITE_API_SECRET in .env. Nothing here can place an order.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser

from fin_agent import config  # noqa: F401  (loads .env, where the Zerodha keys live)
from fin_agent.broker import zerodha


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Zerodha read-only connection.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--login", action="store_true")
    g.add_argument("--holdings", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--logout", action="store_true")
    args = p.parse_args()
    try:
        if args.logout:
            zerodha.logout()
            print("Logged out on this computer.")
            return 0
        if args.login:
            creds = zerodha.Credentials.from_env()
            url = zerodha.login_url(creds.api_key)
            print("1. A Zerodha login page opens in your browser (if not, open this address):")
            print(f"   {url}")
            print("2. Log in with your Zerodha ID, password and 2FA. Type them on Zerodha's page only.")
            print("3. After login the browser goes to your redirect address (the page may say it can't be reached:")
            print("   that's fine). Copy the WHOLE address from the address bar and paste it below.")
            webbrowser.open(url)
            raw = input("Address: ")
            session = zerodha.exchange_token(creds, zerodha.request_token_from(raw))
            zerodha.save_session(session)
            print(f"✅ Logged in as {session['user_name'] or session['user_id']} for today. FinTray can now read "
                  "your holdings (read-only).")
            return 0
        kite = zerodha.connect()
        if args.holdings:
            portfolio = zerodha.holdings_to_portfolio(kite.holdings())
            if not portfolio.holdings:
                print("No holdings in this Zerodha account.")
            for h in portfolio.holdings:
                avg = f"avg ₹{h.avg_cost:,.2f}" if h.avg_cost else "avg ?"
                last = f"last ₹{h.last_price:,.2f}" if h.last_price else "last ?"
                print(f"{h.symbol:15} {h.quantity:>8g} shares  {avg:>16}  {last:>16}")
            for s in portfolio.skipped:
                print(f"skipped {s}")
        else:
            prof = kite.profile()
            print(f"Logged in: {prof.get('user_name', '?')} ({prof.get('broker', 'ZERODHA')})")
            eq = (kite.funds() or {}).get("equity") or {}
            cash = (eq.get("available") or {}).get("live_balance")
            print(f"Equity funds available: ₹{float(cash):,.2f}" if cash is not None else "Funds: not available")
            net = (kite.positions() or {}).get("net") or []
            print(f"Open positions: {sum(1 for x in net if x.get('quantity'))}")
            print(f"Orders today: {len(kite.orders())}")
        return 0
    except zerodha.KiteError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
