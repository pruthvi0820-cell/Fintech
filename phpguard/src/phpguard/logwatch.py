"""Silent detection: read your web server's access log and find attackers without them knowing.

Rules look for well-known PHP attack patterns (webshells in upload folders, PHP stream wrappers and
eval in requests, PHP-CGI argument injection, path traversal, recon of secret files) plus canaries:
paths no real visitor ever opens, so any hit is an attacker exploring your site.

Works on the common "combined" log format (nginx and Apache default).
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from urllib.parse import unquote_plus

from .findings import CRITICAL, HIGH, LOW, MEDIUM, RANK

LINE = re.compile(r'^(?P<ip>\S+) \S+ \S+ \[(?P<time>[^\]]+)\] "(?P<method>[A-Z]+) (?P<target>\S+)[^"]*" '
                  r'(?P<status>\d{3}) ')
DEFAULT_CANARIES = ("/admin-backup.zip", "/old-site/wp-login.php", "/db_backup.sql", "/.aws/credentials")

_UPLOAD_PHP = re.compile(r"/(?:uploads?|images?|img|media|files|assets|wp-content/uploads)/[^?]*\.(?:php\d?|phtml|phar)"
                         r"(?:$|[/?])", re.I)
_INJECTION = re.compile(r"php://|data://|expect://|phar://|zip://|base64_decode\s*\(|\beval\s*\(|\bsystem\s*\(|"
                        r"shell_exec|passthru|<\?php|\bassert\s*\(", re.I)
_CGI_ARGS = re.compile(r"\?(?:-|%2d|%ad)[a-z]", re.I)          # PHP-CGI argument injection (CVE-2012-1823 / 2024-4577)
_TRAVERSAL = re.compile(r"\.\./|\.\.%2f|%2e%2e[/%]", re.I)
_RECON = re.compile(r"/\.env\b|/\.git/|/phpinfo|/info\.php|/vendor/phpunit/|/wp-config\.php\.|/config\.php\.(?:bak|old)"
                    r"|/\.aws/|/server-status|/phpmyadmin|/pma/|/adminer|/\.htpasswd|/backup\.(?:zip|sql|tar)", re.I)

RULES = (
    ("webshell", HIGH, "PHP file requested inside an upload or media folder (possible webshell)",
     lambda path, q: bool(_UPLOAD_PHP.search(path))),
    ("injection", HIGH, "Code injection attempt (PHP wrapper, eval or shell function in the request)",
     lambda path, q: bool(_INJECTION.search(q))),
    ("php-cgi", HIGH, "PHP-CGI argument injection attempt", lambda path, q: bool(_CGI_ARGS.search(q))),
    ("traversal", HIGH, "Path traversal attempt (../)", lambda path, q: bool(_TRAVERSAL.search(q))),
    ("recon", MEDIUM, "Probing for secret or admin files", lambda path, q: bool(_RECON.search(path))),
)
SCANNER_404S = 30


@dataclass
class Attacker:
    ip: str
    severity: str = LOW
    hits: int = 0
    rules: set[str] = field(default_factory=set)
    first: str = ""
    last: str = ""
    samples: list[str] = field(default_factory=list)
    succeeded: bool = False          # a high-risk request got a 200 back: investigate first

    def to_dict(self) -> dict:
        return {"ip": self.ip, "severity": self.severity, "hits": self.hits, "rules": sorted(self.rules),
                "first": self.first, "last": self.last, "samples": self.samples, "succeeded": self.succeeded}


@dataclass
class LogReport:
    lines: int = 0
    unparsed: int = 0
    attackers: list[Attacker] = field(default_factory=list)

    def blocklist(self, min_severity: str = HIGH) -> list[str]:
        return [a.ip for a in self.attackers if RANK[a.severity] >= RANK[min_severity]]


def _raise(a: Attacker, sev: str) -> None:
    if RANK[sev] > RANK[a.severity]:
        a.severity = sev


def classify(target: str, canaries: set[str] | frozenset[str] = frozenset()) -> list[tuple[str, str]]:
    """(rule, severity) for every attack pattern in one request target ("/path?query"). Shared by the log
    watcher and the live firewall, so both agree on what an attack looks like."""
    decoded = unquote_plus(unquote_plus(target))         # attackers double-encode to slip past filters
    path = decoded.split("?", 1)[0]
    hits: list[tuple[str, str]] = []
    if canaries and path.lower().rstrip("/") in canaries:
        hits.append(("canary", CRITICAL))
    hits += [(name, sev) for name, sev, _, test in RULES if test(path, target if name == "php-cgi" else decoded)]
    return hits


def canary_set(canaries) -> frozenset[str]:
    return frozenset(c.lower().rstrip("/") for c in canaries)


def scan(lines, canaries: tuple[str, ...] = DEFAULT_CANARIES) -> LogReport:
    report = LogReport()
    seen: dict[str, Attacker] = {}
    not_found: dict[str, int] = defaultdict(int)
    canaries_norm = canary_set(canaries)

    for raw in lines:
        report.lines += 1
        m = LINE.match(raw)
        if not m:
            report.unparsed += 1
            continue
        ip, target, status = m["ip"], m["target"], int(m["status"])
        hits = classify(target, canaries_norm)
        if status == 404:
            not_found[ip] += 1
        if not hits:
            continue
        a = seen.setdefault(ip, Attacker(ip, first=m["time"]))
        a.hits += 1
        a.last = m["time"]
        for name, sev in hits:
            a.rules.add(name)
            _raise(a, sev)
            if status == 200 and name in ("webshell", "injection", "php-cgi", "traversal"):
                a.succeeded = True
                _raise(a, CRITICAL)
        if len(a.samples) < 3:
            a.samples.append(f"{m['method']} {target[:120]} → {status}")

    for ip, n in not_found.items():
        if n >= SCANNER_404S:
            a = seen.setdefault(ip, Attacker(ip))
            a.rules.add("scanner")
            _raise(a, MEDIUM)
            a.hits = max(a.hits, n)

    report.attackers = sorted(seen.values(), key=lambda a: (-RANK[a.severity], -a.succeeded, -a.hits, a.ip))
    return report


RULE_TEXT = {name: text for name, _, text, _ in RULES} | {
    "canary": "Opened a canary path that no real visitor uses",
    "scanner": f"{SCANNER_404S}+ requests for pages that do not exist (automated scanner)",
}
