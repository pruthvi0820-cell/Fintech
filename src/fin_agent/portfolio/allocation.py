"""Portfolio health: the user's own target mix against the real one, by stock and by sector.

The targets are the user's; FinTray only measures the difference from them. Rupee amounts are the
arithmetic gap to the user's target ("₹X above your target"), never a recommendation to trade.
Targets are saved locally (git-ignored journal folder) so they survive restarts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SECTOR_LIMIT = 0.40          # one sector above this share is flagged (a common rule of thumb, not a law)
DRIFT_FLAG = 0.05            # a holding 5 percentage points or more away from its target is flagged
UNKNOWN_SECTOR = "Unknown"


class TargetError(ValueError):
    """Invalid targets. The message is shown to the user."""


@dataclass(frozen=True)
class Drift:
    symbol: str
    actual: float            # weight, decimal
    target: float | None     # weight, decimal; None = no target set
    value: float             # rupees held now

    @property
    def gap(self) -> float | None:
        return None if self.target is None else self.actual - self.target

    def gap_rupees(self, total: float) -> float | None:
        """Rupees held minus the target share of the total: exact, not from the rounded weight."""
        return None if self.target is None else self.value - self.target * total


def targets_path() -> Path:
    base = os.getenv("FIN_AGENT_JOURNAL_PATH")
    folder = Path(base).parent if base else Path.cwd() / "journal"
    return folder / "targets.json"


def load_targets(path: Path | None = None) -> dict[str, float]:
    path = path or targets_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        return {}                                   # a damaged file must not break the page
    return {str(k).upper(): float(v) for k, v in raw.items() if isinstance(v, (int, float)) and 0 <= v <= 1}


def check_targets(targets: dict[str, float]) -> dict[str, float]:
    """Targets are decimals per symbol. They may add up to less than 100% (the rest is untargeted),
    never more."""
    clean: dict[str, float] = {}
    for sym, t in targets.items():
        if t is None:
            continue
        t = float(t)
        if not 0 <= t <= 1:
            raise TargetError(f"{sym}: a target must be between 0% and 100%.")
        if t > 0:
            clean[sym.strip().upper()] = t
    total = sum(clean.values())
    if total > 1 + 1e-9:
        raise TargetError(f"Your targets add up to {total * 100:.1f}%; they can't be more than 100%.")
    return clean


def save_targets(targets: dict[str, float], path: Path | None = None) -> dict[str, float]:
    clean = check_targets(targets)
    path = path or targets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean, indent=2, sort_keys=True), encoding="utf-8")
    return clean


def drifts(snapshot: dict[str, Any], targets: dict[str, float]) -> list[Drift]:
    """Per holding: actual weight vs target. Uses portfolio_snapshot's weights and total value."""
    values = snapshot.get("values") or {s: w * snapshot["total_value"] for s, w in snapshot["weights"].items()}
    return [Drift(sym, w, targets.get(sym), values[sym]) for sym, w in snapshot["weights"].items()]


def sector_mix(snapshot: dict[str, Any], sectors: dict[str, str | None]) -> dict[str, float]:
    """Weight per sector, largest first. Holdings without a known sector are grouped as Unknown."""
    mix: dict[str, float] = {}
    for sym, w in snapshot["weights"].items():
        sector = sectors.get(sym) or UNKNOWN_SECTOR
        mix[sector] = mix.get(sector, 0.0) + w
    return dict(sorted(mix.items(), key=lambda kv: -kv[1]))


def health_flags(snapshot: dict[str, Any], targets: dict[str, float], mix: dict[str, float]) -> list[str]:
    """Plain facts worth a look. No instructions."""
    flags = list(snapshot.get("concentration_flags", []))
    for sector, w in mix.items():
        if sector != UNKNOWN_SECTOR and w > SECTOR_LIMIT:
            flags.append(f"{sector} is {w * 100:.1f}% of the portfolio, above the {SECTOR_LIMIT * 100:.0f}% "
                         "single-sector guide.")
    if mix.get(UNKNOWN_SECTOR, 0) > 0:
        flags.append(f"The sector of {mix[UNKNOWN_SECTOR] * 100:.1f}% of the portfolio is unknown "
                     "(Yahoo didn't say), so the sector mix is incomplete.")
    total = snapshot["total_value"]
    for d in drifts(snapshot, targets):
        if d.gap is not None and abs(d.gap) >= DRIFT_FLAG:
            side = "above" if d.gap > 0 else "below"
            flags.append(f"{d.symbol} is {d.actual * 100:.1f}% of the portfolio, {abs(d.gap) * 100:.1f} points "
                         f"{side} your {d.target * 100:.1f}% target (about ₹{abs(d.gap_rupees(total)):,.0f}).")
    missing = [s for s in targets if s not in snapshot["weights"]]
    if missing:
        flags.append(f"Targets set for stocks you no longer hold: {', '.join(sorted(missing))}.")
    return flags
