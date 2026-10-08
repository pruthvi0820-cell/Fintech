"""Technical indicators and a trend snapshot, in plain pandas so every formula is inspectable.

Design rule: the trend *label* is decided by an explicit rule in `classify_trend`, not by Claude.
Claude's job is to explain the numbers in words. That keeps the output testable and stops the
model from inventing an RSI value that merely sounds plausible.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252
LOOKBACKS = {"1m": 21, "3m": 63, "6m": 126, "1y": 252}
LARGE_MOVE_THRESHOLD = 0.15

# How many bars back each snapshot field looks. If a large one-day move (usually an unadjusted
# corporate action) falls inside that window, the field is distorted and is removed from the
# snapshot: the model cannot misuse a number it never sees. RSI and MACD use exponential averages
# with long memory, so their windows are approximate (the move's weight has decayed below ~1%).
FIELD_WINDOWS: dict[str, int] = {
    **{f"returns.{k}": n for k, n in LOOKBACKS.items()},
    "sma.20": 20, "sma.50": 50, "sma.200": 200,
    "close_vs_sma_pct.20": 20, "close_vs_sma_pct.50": 50, "close_vs_sma_pct.200": 200,
    "rsi14": 70,
    "macd.macd": 100, "macd.signal": 100, "macd.hist": 100,
    "volatility_annualized.20d": 20, "volatility_annualized.1y": 251,
    "max_drawdown_1y": TRADING_DAYS, "high_52w": TRADING_DAYS, "low_52w": TRADING_DAYS,
    "pct_below_52w_high": TRADING_DAYS,
}
UNRELIABLE_TREND = "unreliable_corporate_action"

# Conventional RSI zones. Stated as text so the model quotes a computed fact instead of judging
# "oversold" itself (the 2026-10-08 audit found RSI 37-46 called oversold three times).
RSI_OVERSOLD, RSI_OVERBOUGHT = 30, 70


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI (the version charting platforms show)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def annualized_vol(close: pd.Series, window: int) -> float | None:
    rets = np.log(close).diff().dropna().tail(window)
    if len(rets) < window:
        return None
    return float(rets.std() * math.sqrt(TRADING_DAYS))


def max_drawdown(close: pd.Series) -> float:
    running_peak = close.cummax()
    return float((close / running_peak - 1).min())


def large_daily_moves(close: pd.Series, threshold: float = LARGE_MOVE_THRESHOLD) -> list[dict[str, Any]]:
    """Single-day moves beyond the threshold.

    For large caps these are usually corporate actions (demerger, bonus, rights) that the data
    vendor did not adjust for, not real crashes. Example: Tata Motors' Oct 2025 demerger into
    TMPV + TMCV. When present, returns and moving averages around that date are distorted.
    """
    chg = close.pct_change().dropna()
    hits = chg[chg.abs() >= threshold]
    return [{"date": ts.date().isoformat(), "change": round(float(v), 4)} for ts, v in hits.items()]


def _bars_since_last_large_move(close: pd.Series, threshold: float = LARGE_MOVE_THRESHOLD) -> int | None:
    """Bars between the newest large one-day move and the last bar (0 = the last bar itself)."""
    chg = close.pct_change().to_numpy()
    hits = np.flatnonzero(np.abs(np.nan_to_num(chg)) >= threshold)
    return None if len(hits) == 0 else len(close) - 1 - int(hits[-1])


def _exclude_distorted(snap: dict[str, Any], bars_ago: int) -> list[str]:
    """Set every field whose window spans the move to None. Returns the removed field paths."""
    removed = []
    for path, window in FIELD_WINDOWS.items():
        if bars_ago > window:
            continue
        *parents, leaf = path.split(".")
        node = snap
        for key in parents:
            node = node[key]
        if node.get(leaf) is not None:
            node[leaf] = None
            removed.append(path)
    if snap["sma"]["50"] is None or snap["sma"]["200"] is None:
        snap["sma50_above_sma200"] = None
        if any(p in removed for p in ("sma.50", "sma.200")):
            snap["trend_label"] = UNRELIABLE_TREND
    return removed


def rsi_zone(value: float | None) -> str | None:
    if value is None:
        return None
    if value < RSI_OVERSOLD:
        return f"oversold (below {RSI_OVERSOLD})"
    if value > RSI_OVERBOUGHT:
        return f"overbought (above {RSI_OVERBOUGHT})"
    return f"neutral ({RSI_OVERSOLD} to {RSI_OVERBOUGHT})"


def ma_order(close: float | None, sma: dict[str, float | None]) -> str | None:
    """Price and moving averages from lowest to highest, e.g. "close < sma20 < sma50 < sma200".

    Names carry no standalone digits, so the numeric check still treats an invented "50" as a claim.
    """
    levels = [("close", close)] + [(f"sma{k}", v) for k, v in sma.items()]
    present = sorted(((v, name) for name, v in levels if v is not None))
    if len(present) < 2:
        return None
    out = present[0][1]
    for (prev, _), (val, name) in zip(present, present[1:]):
        out += f" {'=' if val == prev else '<'} {name}"
    return out


def nearest_levels(close: float | None, levels: dict[str, float | None]
                   ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The closest level strictly above and strictly below the close, as {"name", "value"}.

    None means no level exists on that side in this data (e.g. the close is the 52-week low).
    Computed here because the model, asked for "the nearest level above and below", invented
    "if the close falls below the 200-day" when the close was already below it (2026-10-08 audit).
    """
    if close is None:
        return None, None
    present = [(v, name) for name, v in levels.items() if v is not None]
    above = min(((v, n) for v, n in present if v > close), default=None)
    below = max(((v, n) for v, n in present if v < close), default=None)
    return tuple(None if hit is None else {"name": hit[1], "value": hit[0]} for hit in (above, below))


def _add_derived_facts(snap: dict[str, Any]) -> None:
    """Facts derived from (possibly guarded) values, so a removed input yields None, not a guess."""
    snap["rsi_zone"] = rsi_zone(snap["rsi14"])
    snap["ma_order"] = ma_order(snap["last_close"], snap["sma"])
    levels = {f"sma{k}": v for k, v in snap["sma"].items()}
    levels.update(high_52w=snap["high_52w"], low_52w=snap["low_52w"])
    snap["nearest_level_above"], snap["nearest_level_below"] = nearest_levels(snap["last_close"], levels)
    m = snap["macd"]
    snap["macd_above_signal"] = None if m["macd"] is None or m["signal"] is None else m["macd"] > m["signal"]


def data_warning(snapshot: dict[str, Any]) -> str | None:
    """Plain-language warning for reports. Written by code, so it never depends on the model."""
    dq = snapshot.get("data_quality") or {}
    moves = dq.get("large_daily_moves") or []
    if not moves:
        return None
    listed = ", ".join(f"{m['change'] * 100:+.1f}% on {m['date']}" for m in moves)
    text = (f"**Data warning:** large one-day price move ({listed}). This is usually an unadjusted "
            "corporate action (demerger, bonus, split), not a real crash.")
    removed = dq.get("excluded_fields") or []
    if removed:
        text += f" Figures whose window spans it were removed: {', '.join(removed)}."
    return text


def classify_trend(close: float, sma50: float | None, sma200: float | None) -> str:
    """Classic moving-average stack. Crude on purpose: transparent beats clever."""
    if sma50 is None or sma200 is None:
        return "insufficient_history"
    if close > sma50 > sma200:
        return "uptrend"
    if close < sma50 < sma200:
        return "downtrend"
    return "mixed"


def _last(s: pd.Series) -> float | None:
    v = s.iloc[-1] if len(s) else np.nan
    return None if pd.isna(v) else float(v)


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(x, nd)


def compute_snapshot(bars: pd.DataFrame) -> dict[str, Any]:
    """Reduce a price history to a flat dict of facts. None = not enough data, never a guess."""
    close = bars["Close"].astype(float)
    volume = bars["Volume"].astype(float)
    last_close = float(close.iloc[-1])

    s20, s50, s200 = (_last(sma(close, n)) for n in (20, 50, 200))
    m = macd(close)
    year = close.tail(TRADING_DAYS)

    returns = {
        k: (_r(last_close / float(close.iloc[-n - 1]) - 1) if len(close) > n else None)
        for k, n in LOOKBACKS.items()
    }

    def pct_vs(level: float | None) -> float | None:
        return None if level is None else _r(last_close / level - 1)

    vol20, vol60 = _last(sma(volume, 20)), _last(sma(volume, 60))

    snap = {
        "as_of": bars.index[-1].isoformat(),
        "bars": len(bars),
        "last_close": round(last_close, 4),
        "returns": returns,
        "sma": {"20": _r(s20), "50": _r(s50), "200": _r(s200)},
        "close_vs_sma_pct": {"20": pct_vs(s20), "50": pct_vs(s50), "200": pct_vs(s200)},
        "sma50_above_sma200": None if s50 is None or s200 is None else s50 > s200,
        "rsi14": _r(_last(rsi(close)), 2),
        "macd": {k: _r(_last(m[k])) for k in ("macd", "signal", "hist")},
        "volatility_annualized": {"20d": _r(annualized_vol(close, 20)), "1y": _r(annualized_vol(close, 251))},
        "max_drawdown_1y": _r(max_drawdown(year)),
        "high_52w": round(float(year.max()), 4),
        "low_52w": round(float(year.min()), 4),
        "pct_below_52w_high": _r(last_close / float(year.max()) - 1),
        "volume_ratio_20d_vs_60d": None if not vol20 or not vol60 else _r(vol20 / vol60, 3),
        "trend_label": classify_trend(last_close, s50, s200),
        "data_quality": {"large_daily_moves": large_daily_moves(close), "excluded_fields": []},
    }
    bars_ago = _bars_since_last_large_move(close)
    if bars_ago is not None:
        snap["data_quality"]["excluded_fields"] = _exclude_distorted(snap, bars_ago)
    _add_derived_facts(snap)
    return snap
