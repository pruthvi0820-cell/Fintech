import numpy as np
import pandas as pd
import pytest

from fin_agent.analysis.intraday_backtest import IST, STRATEGIES, run, simulate, summarize


def day_bars(closes, day="2026-10-13", start="09:15", freq="15min", spread=0.2):
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    idx = pd.date_range(f"{day} {start}", periods=len(c), freq=freq, tz=IST)
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) + spread, "Low": np.minimum(o, c) - spread,
                         "Close": c, "Volume": 1e5}, index=idx)


def test_entry_is_next_open_and_target_exit():
    bars = day_bars([100] * 20 + [101, 103, 106, 110, 110])          # 25 candles: 09:15 ... 15:15
    sig = pd.Series(False, index=bars.index)
    sig.iloc[19] = True                                               # signal at 14:00 close
    (t,) = simulate(bars, sig, stop_atr=1.0, reward_risk=2.0, cost_per_side=0.0)
    assert t.entry_time == bars.index[20] and t.entry == bars["Open"].iloc[20]
    assert t.reason == "target" and t.r == pytest.approx(2.0)


def test_square_off_at_1520_and_cutoff():
    bars = day_bars([100.0] * 26)                                     # ends 15:30, flat: neither level
    sig = pd.Series(False, index=bars.index)
    sig.iloc[15] = True                                               # after ATR(14) exists
    (t,) = simulate(bars, sig, cost_per_side=0.0)
    assert t.reason == "squared off 15:20" and t.exit_time.tz_convert(IST).strftime("%H:%M") == "15:30"
    late = pd.Series(False, index=bars.index)
    late.iloc[23] = True                                              # 15:00 signal -> 15:15 entry: past cut-off
    assert simulate(bars, late) == []


def test_gap_stop_and_stop_first_rule():
    bars = day_bars([100] * 20 + [100, 90, 95])
    bars.iloc[21, bars.columns.get_loc("Open")] = 91.0                # candle 21 OPENS far below the stop
    bars.iloc[21, bars.columns.get_loc("High")] = 91.2
    sig = pd.Series(False, index=bars.index)
    sig.iloc[19] = True
    (t,) = simulate(bars, sig, stop_atr=1.0, cost_per_side=0.0)
    assert t.reason == "stop (gap)" and t.exit == 91.0 and t.r < -1     # the gap made it worse than -1R


def test_no_signal_uses_the_future():
    rng = np.random.default_rng(2)
    bars = day_bars(100 + rng.normal(0, 0.4, 25).cumsum())
    for name, (_, rule) in STRATEGIES.items():
        changed = bars.copy()
        changed.iloc[15:, :4] *= 1.3
        assert rule(bars).iloc[:15].equals(rule(changed).iloc[:15]), name


def test_random_prices_show_no_edge():
    rng = np.random.default_rng(4)
    frames = []
    for d in pd.bdate_range("2026-01-01", periods=60):
        frames.append(day_bars(100 * np.exp(rng.normal(0, 0.003, 25).cumsum()), day=d.date().isoformat()))
    bars = pd.concat(frames)
    out = run(bars, "breakout", cost_per_side=0.0)
    s, b = out["result"], out["baseline"]
    assert b["trades"] > 80 and abs(b["avg_r"]) < 0.15                 # entering anywhere: about 0R before costs
    assert s["avg_r"] - 2 * s["se_r"] < 0.3                             # the strategy can't look clearly better
    assert not out["verdict"].startswith("Beat both")


def test_summary_and_errors():
    assert summarize([])["trades"] == 0
    with pytest.raises(ValueError, match="Unknown strategy"):
        run(day_bars([100] * 80), "magic")
    with pytest.raises(ValueError, match="Not enough candles"):
        run(day_bars([100] * 10), "breakout")
