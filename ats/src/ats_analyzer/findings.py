"""One shape for every problem the parser notices, so later scoring can weigh them."""

from dataclasses import asdict, dataclass

HIGH, MEDIUM, LOW = "high", "medium", "low"


@dataclass(frozen=True)
class Finding:
    code: str          # stable id, e.g. "pdf_no_text"; scoring and tests key on this, not the message
    severity: str      # high: an ATS likely loses data; medium: may misread; low: worth fixing
    message: str

    def to_dict(self) -> dict:
        return asdict(self)
