"""Stock summary: measured records for each trading style, the minimum to buy, and the user's own record.

Everything here is measured from the past. Nothing is a forecast, and the page says so.

Holding-period records answer "if I had bought on any day and held for N trading days, how often
would I have made money after costs, and how much?". Start days overlap (buying on Monday or
Tuesday and holding a year share almost the whole year), so `independent` reports roughly how many
non-overlapping periods the data holds: that is the honest sample size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from fin_agent.analysis.backtest import BacktestResult
from fin_agent.analysis.indicators import _bars_since_last_large_move

# Label -> trading days held. Long ones need a long price history (fetched separately, all available).
HORIZONS: dict[str, int] = {"1 week": 5, "1 month": 21, "3 months": 63, "1 year": 252, "3 years": 756,
                            "5 years": 1260}
MIN_INDEPENDENT = 3      # fewer non-overlapping periods than this is flagged as thin evidence


@dataclass(frozen=True)
class Record:
    style: str
    based_on: str                  # what the numbers come from, in words
    samples: int                   # trades, or start days for holding periods
    independent: int               # non-overlapping periods (== samples for trades)
    win_rate: float | None         # share ending above zero after costs
    avg_win: float | None          # mean return of the winners (decimal)
    avg_loss: float | None         # mean return of the losers (decimal, <= 0)
    holding: str                   # typical time held, in words
    note: str | None = None

    @property
    def loss_rate(self) -> float | None:
        return None if self.win_rate is None else 1 - self.win_rate


def _usable_close(close: pd.Series) -> tuple[pd.Series, str | None]:
    """Drop the bars before an unadjusted corporate-action gap, like the backtest does."""
    close = close.dropna().astype(float)
    bars_ago = _bars_since_last_large_move(close)
    if bars_ago is None:
        return close, None
    kept = close.iloc[len(close) - 1 - bars_ago:]
    return kept, (f"Only the {len(kept)} trading days after a large one-day move (likely an unadjusted "
                  "corporate action) are used.")


def holding_record(close: pd.Series, label: str, days: int, cost_per_side: float) -> Record:
    """Buy at a day's close, sell `days` trading days later at the close, costs on both sides."""
    close, gap_note = _usable_close(close)
    exit_ = close.shift(-days)
    net = ((exit_ * (1 - cost_per_side)) / (close * (1 + cost_per_side)) - 1).dropna()
    independent = len(close) // days if days else 0
    based_on = f"every start day in the last {len(close) / 252:.1f} years of prices"
    notes = [gap_note] if gap_note else []
    if net.empty:
        need = days / 252
        return Record(f"Hold {label}", based_on, 0, independent, None, None, None, label,
                      note=" ".join([f"Not enough history: needs more than {need:.1f} years of prices.", *notes]))
    if independent < MIN_INDEPENDENT:
        notes.insert(0, f"Thin evidence: the data holds only {independent} separate {label} period"
                        f"{'' if independent == 1 else 's'}, so this record could easily change.")
    wins, losses = net[net > 0], net[net <= 0]
    return Record(
        style=f"Hold {label}", based_on=based_on, samples=len(net), independent=independent,
        win_rate=float(len(wins) / len(net)),
        avg_win=float(wins.mean()) if len(wins) else None,
        avg_loss=float(losses.mean()) if len(losses) else None,
        holding=label, note=" ".join(notes) or None,
    )


def swing_record(result: BacktestResult) -> Record:
    trades = result.trades
    if not trades:
        return Record("Swing (FinTray's rules)", "backtest of the buy/sell conditions", 0, 0, None, None, None,
                      "–", note="The rules made no trades in this period.")
    wins = [t.net_return for t in trades if t.net_return > 0]
    losses = [t.net_return for t in trades if t.net_return <= 0]
    avg_days = sum(t.bars_held for t in trades) / len(trades)
    return Record(
        style="Swing (FinTray's rules)", based_on=f"backtest, {result.start} to {result.end}",
        samples=len(trades), independent=len(trades), win_rate=len(wins) / len(trades),
        avg_win=sum(wins) / len(wins) if wins else None,
        avg_loss=sum(losses) / len(losses) if losses else None,
        holding=f"{avg_days:.0f} trading days on average",
        note=result.notes[0] if result.notes else None,
    )


def intraday_record() -> Record:
    return Record("Intraday (same day)", "minute-by-minute prices", 0, 0, None, None, None, "minutes to hours",
                  note="Needs minute-by-minute prices, which come with the Upstox connection (a later step).")


def minimum_to_buy(price: float, cost_per_side: float) -> dict[str, float]:
    """One share is the minimum for delivery (cash) buying on NSE; costs are the user's estimate."""
    costs = price * cost_per_side
    return {"shares": 1, "price": round(price, 2), "costs": round(costs, 2), "total": round(price + costs, 2)}


def your_record(entries: pd.DataFrame, ticker: str) -> dict[str, Any]:
    """The user's own journal for this stock."""
    if entries is None or entries.empty:
        return {"closed": 0, "open": 0, "skipped": 0, "wins": 0, "losses": 0, "pnl": 0.0, "win_rate": None}
    mine = entries[entries["ticker"].str.upper() == ticker.upper()]
    closed = mine[mine["status"] == "closed"]
    wins = int((closed["pnl"] > 0).sum())
    return {
        "closed": len(closed), "open": int((mine["status"] == "open").sum()),
        "skipped": int((mine["status"] == "skipped").sum()), "wins": wins, "losses": len(closed) - wins,
        "pnl": round(float(closed["pnl"].sum()), 2) if len(closed) else 0.0,
        "win_rate": wins / len(closed) if len(closed) else None,
    }


def stock_records(result: BacktestResult, long_close: pd.Series | None, cost_per_side: float) -> list[Record]:
    """All styles in display order. `long_close` is the longest close history available; None if it couldn't be fetched."""
    records = [intraday_record(), swing_record(result)]
    for label, days in HORIZONS.items():
        if long_close is None or long_close.empty:
            records.append(Record(f"Hold {label}", "long price history", 0, 0, None, None, None, label,
                                  note="The long price history could not be fetched."))
        else:
            records.append(holding_record(long_close, label, days, cost_per_side))
    return records
