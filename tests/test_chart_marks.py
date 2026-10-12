import json

import numpy as np
import pandas as pd
import pytest

from fin_agent.analysis.chart_marks import MarkError, add_mark, delete_mark, load_marks, mark_alerts
from fin_agent.pipelines.alerts import gather_alerts


def bars(closes):
    idx = pd.bdate_range("2026-10-01", periods=len(closes), tz="UTC")
    c = np.array(closes, dtype=float)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1e6}, index=idx)


@pytest.fixture
def path(tmp_path):
    return tmp_path / "chart_marks.json"


def test_add_load_and_delete(path):
    add_mark("tcs.ns", 3500, "  resistance  ", path)
    add_mark("TCS.NS", 3200.456, "", path)
    marks = load_marks(path)["TCS.NS"]
    assert [(m.price, m.label) for m in marks] == [(3200.46, "My mark"), (3500.0, "resistance")]
    assert delete_mark("TCS.NS", 3500, path) and not delete_mark("TCS.NS", 3500, path)
    assert [m.price for m in load_marks(path)["TCS.NS"]] == [3200.46]
    delete_mark("TCS.NS", 3200.46, path)
    assert load_marks(path) == {} and json.loads(path.read_text()) == {}


@pytest.mark.parametrize("ticker, price, label, msg", [
    ("", 10, "", "Pick a stock"),
    ("A.NS", 0, "", "above 0"),
    ("A.NS", float("nan"), "", "above 0"),
    ("A.NS", "abc", "", "must be a number"),
    ("A.NS", 10, "x" * 41, "under 40"),
])
def test_bad_input_is_refused(path, ticker, price, label, msg):
    with pytest.raises(MarkError, match=msg):
        add_mark(ticker, price, label, path)


def test_duplicates_and_limit(path, monkeypatch):
    add_mark("A.NS", 100, "x", path)
    with pytest.raises(MarkError, match="already has a mark at 100.00"):
        add_mark("A.NS", 100.001, "y", path)
    monkeypatch.setattr("fin_agent.analysis.chart_marks.MAX_MARKS_PER_STOCK", 1)
    with pytest.raises(MarkError, match="already has 1 marks"):
        add_mark("A.NS", 200, "y", path)


@pytest.mark.parametrize("content", ["not json", "[1, 2]", '{"A.NS": "x"}',
                                     '{"A.NS": [{"price": "bad"}, {"label": "no price"}, {"price": -5}]}'])
def test_damaged_file_means_no_marks(path, content):
    path.write_text(content)
    assert load_marks(path) == {}


def test_alerts_crossed_up_down_and_near(path):
    add_mark("UP.NS", 105, "breakout", path)
    add_mark("DN.NS", 95, "support", path)
    add_mark("NR.NS", 101, "near", path)
    add_mark("FAR.NS", 150, "far", path)
    data = {"UP.NS": bars([100, 106]), "DN.NS": bars([100, 94]), "NR.NS": bars([99, 100]), "FAR.NS": bars([99, 100])}
    alerts = {a.ticker: a for a in mark_alerts(load_marks(path), data)}
    assert set(alerts) == {"UP.NS", "DN.NS", "NR.NS"}
    assert alerts["UP.NS"].level == "high" and "up through your mark 'breakout' ₹105.00" in alerts["UP.NS"].text
    assert "down through your mark 'support'" in alerts["DN.NS"].text
    assert alerts["NR.NS"].level == "info" and "1.0% from your mark 'near'" in alerts["NR.NS"].text


def test_alerts_skip_missing_or_short_data(path):
    add_mark("A.NS", 100, "x", path)
    assert mark_alerts(load_marks(path), {"A.NS": None}) == []
    assert mark_alerts(load_marks(path), {"A.NS": bars([100])}) == []


def test_gather_alerts_checks_marked_stocks_without_watching_them(path):
    add_mark("UP.NS", 105, "breakout", path)
    alerts, tickers = gather_alerts([], set(), pd.Timestamp("2026-10-02").date(), lambda tk: bars([100, 106]),
                                    marks=load_marks(path))
    assert tickers == [] and [a.kind for a in alerts] == ["mark"]


def test_marks_are_drawn_on_the_chart(path):
    pytest.importorskip("plotly")
    from fin_agent.charts import DRAW_CONFIG, candle_chart
    add_mark("A.NS", 1234.5, "resistance", path)
    rng = np.random.default_rng(1)
    fig = candle_chart(bars(list(1000 + rng.normal(0, 5, 60).cumsum())), "A.NS", marks=load_marks(path)["A.NS"])
    assert any(a.text == "✏️ resistance 1,234.50" for a in fig.layout.annotations)
    assert "drawline" in DRAW_CONFIG["modeBarButtonsToAdd"] and fig.layout.newshape.line.color
