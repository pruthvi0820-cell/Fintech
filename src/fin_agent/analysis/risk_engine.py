"""Risk engine: hard rules every new trade must pass. Python only; the AI cannot change or skip them.

Checked before a paper trade opens (and, in Step 7, before any real order is even shown for
approval):
- a stop-loss is set, on the correct side of the entry;
- the trade risks at most `max_risk_pct` of capital (loss if the stop is hit, with costs);
- reward-to-risk is at least `min_reward_risk`;
- today: fewer than `max_losses_per_day` losing trades (the "2-loss stop"), fewer than
  `max_trades_per_day` trades, and today's losses plus this trade's risk stay within
  `daily_loss_pct` of capital;
- India: NSE stocks only during market hours, no new intraday trade after the cut-off (brokers
  square off intraday positions around 15:20); rupee currency pairs during NSE currency hours.

Your own limits live in journal/risk_rules.json. Making a rule STRICTER applies at once; making
it LOOSER applies from tomorrow, so a bad day can't talk you into raising your own limits. Values
can never go past the hard ceilings below.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fin_agent.analysis.alerts import watchlist_path

IST = ZoneInfo("Asia/Kolkata")
NSE_OPEN, NSE_CLOSE = time(9, 15), time(15, 30)
CURRENCY_OPEN, CURRENCY_CLOSE = time(9, 0), time(17, 0)

# The loosest any rule may ever be (and whether a bigger number is looser).
CEILINGS = {"max_risk_pct": (0.02, True), "daily_loss_pct": (0.05, True), "max_losses_per_day": (3, True),
            "max_trades_per_day": (10, True), "min_reward_risk": (1.0, False)}


class RiskRuleError(ValueError):
    """A rule value is invalid. The message is shown to the user."""


@dataclass(frozen=True)
class RiskRules:
    max_risk_pct: float = 0.01          # of capital, per trade
    daily_loss_pct: float = 0.02        # of capital, per day (realised losses + this trade's risk)
    max_losses_per_day: int = 2         # the 2-loss stop
    max_trades_per_day: int = 5
    min_reward_risk: float = 1.5
    intraday_cutoff: str = "15:00"      # no new intraday trades after this (IST)

    def validate(self) -> "RiskRules":
        for name, (ceiling, bigger_is_looser) in CEILINGS.items():
            v = getattr(self, name)
            if v is None or v <= 0:
                raise RiskRuleError(f"{name} must be above 0.")
            if (v > ceiling) if bigger_is_looser else (v < ceiling):
                raise RiskRuleError(f"{name} can't be {'above' if bigger_is_looser else 'below'} {ceiling} "
                                    "(FinTray's hard limit).")
        try:
            cut = time.fromisoformat(self.intraday_cutoff)
        except ValueError:
            raise RiskRuleError("The intraday cut-off must be a time like 15:00.") from None
        if not NSE_OPEN < cut <= time(15, 15):
            raise RiskRuleError("The intraday cut-off must be between 09:15 and 15:15.")
        return self

    def looser_than(self, other: "RiskRules") -> list[str]:
        """Names of rules where self is looser than other."""
        out = []
        for name, (_, bigger_is_looser) in CEILINGS.items():
            a, b = getattr(self, name), getattr(other, name)
            if (a > b) if bigger_is_looser else (a < b):
                out.append(name)
        if time.fromisoformat(self.intraday_cutoff) > time.fromisoformat(other.intraday_cutoff):
            out.append("intraday_cutoff")
        return out


@dataclass
class Verdict:
    allowed: bool
    blocks: list[str] = field(default_factory=list)     # why it is refused (any block refuses)
    notes: list[str] = field(default_factory=list)      # status lines shown either way

    def message(self) -> str:
        return " ".join(self.blocks) if self.blocks else "All risk rules passed."


# ---------------------------------------------------------------- rules file

def rules_path() -> Path:
    return watchlist_path().parent / "risk_rules.json"


def load_rules(path: Path | None = None, today: date | None = None) -> RiskRules:
    """Today's rules: the saved ones, with any loosening that is due by today applied."""
    today = today or datetime.now(IST).date()
    try:
        raw = json.loads((path or rules_path()).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return RiskRules()
    names = {f.name for f in fields(RiskRules)}
    try:
        current = RiskRules(**{k: v for k, v in raw.get("current", {}).items() if k in names}).validate()
        pending = raw.get("pending")
        if pending and date.fromisoformat(pending["from"]) <= today:
            current = RiskRules(**{k: v for k, v in pending["rules"].items() if k in names}).validate()
    except (RiskRuleError, TypeError, ValueError, KeyError):
        return RiskRules()                       # a damaged file falls back to the safe defaults
    return current


def save_rules(new: RiskRules, path: Path | None = None, today: date | None = None) -> str:
    """Stricter values apply now; looser ones from tomorrow. Returns what happened, in words."""
    new.validate()
    today = today or datetime.now(IST).date()
    path = path or rules_path()
    current = load_rules(path, today)
    looser = new.looser_than(current)
    if looser:
        # Tighten now whatever is stricter; schedule the full new set for tomorrow.
        now_rules = RiskRules(**{f.name: (getattr(current, f.name) if f.name in looser else getattr(new, f.name))
                                 for f in fields(RiskRules)})
        data = {"current": asdict(now_rules), "pending": {"from": (today + timedelta(days=1)).isoformat(),
                                                          "rules": asdict(new)}}
        msg = (f"Saved. Looser limits ({', '.join(looser)}) start tomorrow, {today + timedelta(days=1):%d %b}; "
               "stricter ones apply now.")
    else:
        data = {"current": asdict(new)}
        msg = "Saved. The new limits apply now."
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return msg


# ---------------------------------------------------------------- the check

def _market_for(ticker: str) -> str:
    t = ticker.upper()
    if t.endswith("INR=X"):
        return "currency"
    if t.endswith((".NS", ".BO")) or t.startswith("^NSE") or t == "^BSESN":
        return "nse"
    return "other"


def check_trade(*, ticker: str, entry: float, stop: float, target: float, shares: int, capital: float,
                cost_per_side: float, day_results: list[float], trades_today: int, rules: RiskRules,
                now: datetime | None = None, intraday: bool = False) -> Verdict:
    """day_results: P&L of trades closed today (any sign). trades_today: trades opened today.
    `now` (with timezone) enables the market-hours checks; leave it None for a past-dated paper trade."""
    v = Verdict(True)
    block = v.blocks.append
    if not stop or stop <= 0 or stop >= entry:
        block("Set a stop-loss below the entry price: every trade needs one.")
    if target <= entry:
        block("The target must be above the entry price.")
    if shares < 1 or capital <= 0:
        block("Shares and capital must be above 0.")
    if v.blocks:
        v.allowed = False
        return v

    risk = shares * ((entry - stop) + cost_per_side * (entry + stop))
    rr = (target - entry) / (entry - stop)
    if risk > rules.max_risk_pct * capital + 1e-6:
        block(f"This trade risks ₹{risk:,.0f} ({risk / capital * 100:.2f}% of ₹{capital:,.0f}); your limit is "
              f"{rules.max_risk_pct * 100:g}% (₹{rules.max_risk_pct * capital:,.0f}). Use fewer shares or a "
              "closer stop.")
    if rr < rules.min_reward_risk - 1e-9:
        block(f"Reward-to-risk is {rr:.2f}; your minimum is {rules.min_reward_risk:g}.")

    losses = [p for p in day_results if p < 0]
    lost = -sum(losses)
    limit = rules.daily_loss_pct * capital
    if len(losses) >= rules.max_losses_per_day:
        block(f"{len(losses)} losing trades today: your {rules.max_losses_per_day}-loss stop is on. No more "
              "trades today. Step away and review them in the journal.")
    if trades_today >= rules.max_trades_per_day:
        block(f"{trades_today} trades today already; your limit is {rules.max_trades_per_day}.")
    if lost >= limit - 1e-6:
        block(f"Today's losses ₹{lost:,.0f} reached your daily limit ₹{limit:,.0f}. Trading is over for today.")
    elif lost + risk > limit + 1e-6:
        block(f"If this trade hits its stop, today's loss would be ₹{lost + risk:,.0f}, over your daily limit "
              f"₹{limit:,.0f}. Use fewer shares (at most ₹{limit - lost:,.0f} of risk is left today).")

    if now is not None:
        t = now.astimezone(IST)
        market = _market_for(ticker)
        if market in ("nse", "currency") and t.weekday() >= 5:
            block("Indian markets are closed on weekends.")
        elif market == "nse" and not NSE_OPEN <= t.time() <= NSE_CLOSE:
            block("NSE is closed now (open 09:15–15:30 IST, Monday to Friday).")
        elif market == "currency" and not CURRENCY_OPEN <= t.time() <= CURRENCY_CLOSE:
            block("NSE currency trading is closed now (09:00–17:00 IST).")
        elif intraday and market == "nse" and t.time() > time.fromisoformat(rules.intraday_cutoff):
            block(f"No new intraday trades after {rules.intraday_cutoff} IST: brokers close intraday positions "
                  "around 15:20, leaving no room for the trade to work.")
        v.notes.append("Exchange holidays are not known to FinTray: check NSE's holiday list.")

    v.notes.append(f"Today: {len(losses)} of {rules.max_losses_per_day} losses, {trades_today} of "
                   f"{rules.max_trades_per_day} trades, ₹{lost:,.0f} of ₹{limit:,.0f} daily loss limit used.")
    v.allowed = not v.blocks
    return v


def rules_table(rules: RiskRules) -> list[dict[str, Any]]:
    return [
        {"Rule": "Max risk per trade", "Your limit": f"{rules.max_risk_pct * 100:g}% of capital"},
        {"Rule": "Daily loss limit", "Your limit": f"{rules.daily_loss_pct * 100:g}% of capital"},
        {"Rule": "Stop after this many losses in a day", "Your limit": str(rules.max_losses_per_day)},
        {"Rule": "Max trades per day", "Your limit": str(rules.max_trades_per_day)},
        {"Rule": "Minimum reward-to-risk", "Your limit": f"{rules.min_reward_risk:g}"},
        {"Rule": "No new intraday trades after", "Your limit": f"{rules.intraday_cutoff} IST"},
    ]
