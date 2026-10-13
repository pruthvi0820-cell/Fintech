"""MetaTrader 5 (XM demo) read-only check from the command line.

Usage (Windows, with the MT5 program open and logged in to your XM DEMO account):
    pip install -e ".[mt5]"                     # once
    python scripts/mt5.py --check               # account, clock offset, last GOLD candles
    python scripts/mt5.py --positions           # open trades with the 1% risk check
    python scripts/mt5.py --trades              # closed trades of the last 30 days and stats

Your MT5 password stays in the MT5 program. Demo accounts only; nothing here can place a trade.
"""

from __future__ import annotations

import argparse
import sys

from fin_agent import config  # noqa: F401  (loads .env)
from fin_agent.broker import mt5


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="MetaTrader 5 (XM demo), read-only.")
    g = p.add_mutually_exclusive_group(required=True)
    for flag in ("--check", "--positions", "--trades"):
        g.add_argument(flag, action="store_true")
    p.add_argument("--symbol", default="GOLD", help="MT5 symbol for --check (default GOLD)")
    args = p.parse_args()
    try:
        lib = mt5.connect()
        if args.check:
            a = mt5.account(lib)
            print(f"Connected: {a['name']} · login {a['login']} · {a['server']} (demo ✅)")
            print(f"Balance {a['balance']:,.2f} {a['currency']} · equity {a['equity']:,.2f}")
            hours, how = mt5.utc_offset_hours(lib, args.symbol.upper())
            print(f"Server clock: UTC{hours:+g} ({how})")
            df = mt5.candles(args.symbol, "15m", lib=lib)
            print(f"{len(df)} fifteen-minute {args.symbol.upper()} candles; the last three (IST):")
            for t, r in df.tail(3).iterrows():
                print(f"  {t:%d %b %H:%M}  O {r.Open:,.2f}  H {r.High:,.2f}  L {r.Low:,.2f}  C {r.Close:,.2f}")
            print("✅ MT5 connection works (read-only).")
        elif args.positions:
            ps = mt5.positions(lib)
            if not ps:
                print("No open trades.")
            for x in ps:
                warn = ("  ⚠️ " + "; ".join(x["flags"])) if x["flags"] else ""
                sl = f"{x['sl']:,.2f}" if x["sl"] else "none"
                print(f"#{x['ticket']} {x['side']} {x['lots']:g} {x['symbol']} @ {x['entry']:,.2f}  SL {sl}  "
                      f"now {x['now']:,.2f}  P/L {x['profit']:+,.2f}{warn}")
        else:
            trades = mt5.closed_trades(lib=lib)
            for t in trades:
                r = f"  {t['r']:+.2f}R" if t["r"] is not None else ""
                print(f"{t['closed']:%d %b %H:%M} {t['side']} {t['lots']:g} {t['symbol']} {t['entry']:,.2f} -> "
                      f"{t['exit']:,.2f}  P/L {t['profit']:+,.2f}{r}")
            s = mt5.stats(trades)
            if s["trades"]:
                note = "  (too few to judge: 20+ needed)" if s["too_few"] else ""
                print(f"{s['trades']} trades · win rate {s['win_rate']:.0%} · expectancy {s['expectancy']:+,.2f} per "
                      f"trade · total {s['total']:+,.2f}{note}")
            else:
                print("No closed trades in the last 30 days.")
        return 0
    except mt5.MT5Error as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
