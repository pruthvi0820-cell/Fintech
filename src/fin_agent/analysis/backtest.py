"""Backtest of the swing rules in signals.py on one stock's past daily candles.

Purpose: before a user acts on "4 of 5 buy conditions met", show how acting on exactly that rule
would have gone on this stock. It is evidence about the past, not a forecast.

Rules (long only, one position at a time):
- signal is read at a day's close; the trade happens at the NEXT day's open (no look-ahead)
- enter when buy_score >= entry_score
- exit when buy_score <= exit_score, or after max_hold trading days
- cost_per_side covers STT, brokerage, charges and slippage, as a fraction of the trade value
If a corporate-action gap is in the data, only the bars after it are tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from fin_agent.analysis.indicators import _bars_since_last_large_move
from fin_agent.analysis.signals import signal_frame

MIN_TRADES_TO_JUDGE = 10
DEFAULT_COST_PER_SIDE = 0.0015    # rough: delivery STT 0.1% + charges + slippage; adjust to your broker


@dataclass(frozen=True)
class Trade:
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    bars_held: int
    net_return: float        # after costs, decimal
    exit_reason: str         # "signal" | "max_hold" | "end_of_data"


@dataclass
class BacktestResult:
    start: str
    end: str
    trades: list[Trade] = field(default_factory=list)
    total_return: float = 0.0           # compounded, after costs
    buy_hold_return: float = 0.0        # same period, open of first tested day to last close
    max_drawdown: float = 0.0           # of the strategy's daily equity
    exposure: float = 0.0               # share of days holding a position
    params: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float | None:
        return None if not self.trades else sum(t.net_return > 0 for t in self.trades) / self.n_trades

    @property
    def avg_return(self) -> float | None:
        return None if not self.trades else sum(t.net_return for t in self.trades) / self.n_trades


def backtest(
    bars: pd.DataFrame,
    entry_score: int = 4,
    exit_score: int = 2,
    max_hold: int = 20,
    cost_per_side: float = DEFAULT_COST_PER_SIDE,
) -> BacktestResult:
    if not 0 <= exit_score < entry_score <= 5:
        raise ValueError("Need 0 <= exit_score < entry_score <= 5.")
    if max_hold < 1 or not 0 <= cost_per_side < 0.05:
        raise ValueError("max_hold must be >= 1 and cost_per_side between 0 and 0.05.")

    notes: list[str] = []
    gap = _bars_since_last_large_move(bars["Close"].astype(float))
    if gap is not None:
        bars = bars.iloc[len(bars) - gap:]
        notes.append("A corporate-action gap was found; only the days after it were tested.")

    f = signal_frame(bars)
    tested = f[f["buy_score"].notna()]
    params = {"entry_score": entry_score, "exit_score": exit_score, "max_hold": max_hold,
              "cost_per_side": cost_per_side}
    if len(tested) < 2:
        return BacktestResult(start="-", end="-", params=params,
                              notes=notes + ["Not enough history to test."])

    dates = [d.date().isoformat() for d in tested.index]
    o, c = tested["open"].to_numpy(), tested["close"].to_numpy()
    score = tested["buy_score"].to_numpy()

    trades: list[Trade] = []
    equity, peak, max_dd, days_in = 1.0, 1.0, 0.0, 0
    entry_i: int | None = None
    pending: str | None = None            # "enter" / "exit:<reason>" decided at a close, done next open

    for i in range(len(tested)):
        # 1) act at today's open on yesterday's decision
        if pending == "enter":
            entry_i = i
            equity *= c[i] / (o[i] * (1 + cost_per_side))
        elif pending and pending.startswith("exit") and entry_i is not None:
            equity *= o[i] * (1 - cost_per_side) / c[i - 1]
            trades.append(_trade(dates, o, entry_i, i, o[i], cost_per_side, pending.split(":")[1]))
            entry_i = None
        elif entry_i is not None and entry_i < i:
            equity *= c[i] / c[i - 1]
        pending = None
        days_in += entry_i is not None

        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1)

        # 2) decide at today's close for tomorrow's open
        if i == len(tested) - 1:
            break
        if entry_i is None and score[i] >= entry_score:
            pending = "enter"
        elif entry_i is not None:
            if score[i] <= exit_score:
                pending = "exit:signal"
            elif i - entry_i + 1 >= max_hold:
                pending = "exit:max_hold"

    if entry_i is not None:   # still open at the end: valued at the last close, after selling costs
        last = len(tested) - 1
        equity *= 1 - cost_per_side
        trades.append(_trade(dates, o, entry_i, last, c[last], cost_per_side, "end_of_data"))
        notes.append("The last trade was still open; it is valued at the last close.")

    if len(trades) < MIN_TRADES_TO_JUDGE:
        notes.append(f"Only {len(trades)} trades: too few to judge whether the rule works on this stock.")

    return BacktestResult(
        start=dates[0], end=dates[-1], trades=trades,
        total_return=equity - 1,
        buy_hold_return=c[-1] / o[0] - 1,
        max_drawdown=max_dd,
        exposure=days_in / len(tested),
        params=params, notes=notes,
    )


def _trade(dates: list[str], o: Any, entry_i: int, exit_i: int, exit_price: float,
           cost: float, reason: str) -> Trade:
    entry = float(o[entry_i])
    return Trade(dates[entry_i], round(entry, 2), dates[exit_i], round(float(exit_price), 2),
                 exit_i - entry_i, (exit_price * (1 - cost)) / (entry * (1 + cost)) - 1, reason)
