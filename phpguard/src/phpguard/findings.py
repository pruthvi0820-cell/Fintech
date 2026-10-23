"""One shape for every problem PHPGuard reports, so audits, site checks and log alerts share a report."""

from dataclasses import asdict, dataclass

CRITICAL, HIGH, MEDIUM, LOW, INFO = "critical", "high", "medium", "low", "info"
RANK = {CRITICAL: 4, HIGH: 3, MEDIUM: 2, LOW: 1, INFO: 0}


@dataclass(frozen=True)
class Finding:
    area: str          # patch | harden | hide | exposure | detect
    severity: str
    title: str
    detail: str
    fix: str
    host: str = ""
    refs: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["refs"] = list(self.refs)
        return d
