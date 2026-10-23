"""Is this PHP build exposed to a known vulnerability, or out of support?

The advisory file (advisories.json) lists, per vulnerability, the first fixed release of each PHP
branch. Add an entry when the next PHP2Shell happens; no code changes needed.

Linux distributions often patch PHP *without* changing its upstream version ("8.1.2-1ubuntu2.18" can
contain the fix). A distro build is therefore never declared vulnerable from its version alone: it is
flagged for a check against that distro's security tracker.
"""

import json
import re
from datetime import date
from pathlib import Path

from .findings import CRITICAL, HIGH, INFO, LOW, Finding

DEFAULT_FILE = Path(__file__).with_name("advisories.json")
_VERSION = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")
_DISTRO = re.compile(r"ubuntu|~deb|deb\d|\+deb|\.el\d|\+el\d|module\+el|amzn|suse|\.fc\d|-r\d+$|sury", re.I)
TRACKERS = ("https://ubuntu.com/security/{cve}", "https://security-tracker.debian.org/tracker/{cve}",
            "https://access.redhat.com/security/cve/{cve}")
STALE_DAYS = 30


def load(path: str | Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_FILE
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"cannot read advisory file {p}: {e}") from e
    if not isinstance(data.get("advisories"), list) or not isinstance(data.get("branches"), dict):
        raise ValueError(f"advisory file {p} is missing 'advisories' or 'branches'")
    return data


def parse_version(text: str | None) -> tuple[int, int, int] | None:
    m = _VERSION.search(text or "")
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)) if m else None


def is_distro_build(package_version: str | None) -> bool:
    return bool(package_version and _DISTRO.search(package_version))


def _fmt(v: tuple[int, int, int]) -> str:
    return ".".join(map(str, v))


def assess(php_version: str | None, package_version: str | None = None, host: str = "",
           data: dict | None = None, today: date | None = None) -> list[Finding]:
    data = data or load()
    today = today or date.today()
    out: list[Finding] = []

    try:
        updated = date.fromisoformat(data.get("updated", ""))
        if (today - updated).days > STALE_DAYS:
            out.append(Finding("patch", LOW, "Advisory data may be out of date",
                               f"advisories.json was last updated {updated}", "Add new PHP advisories to "
                               "advisories.json, or pull the latest PHPGuard", host))
    except ValueError:
        pass

    v = parse_version(package_version) or parse_version(php_version)
    if v is None:
        out.append(Finding("patch", HIGH, "PHP version unknown", "could not read the PHP version on this host",
                           "Run `phpguard collect` on the server, or record the version in the inventory", host))
        return out

    branch = f"{v[0]}.{v[1]}"
    distro = is_distro_build(package_version)
    shown = package_version or _fmt(v)

    for adv in data["advisories"]:
        start = parse_version(adv.get("affects_from")) or (0, 0, 0)
        if v < start:
            continue
        cves = adv.get("cves", [])
        refs = tuple(adv.get("references", []))
        fixed = adv.get("fixed", {}).get(branch)
        fixed_v = parse_version(fixed)
        newest_fixed = max((parse_version(b) or (0, 0, 0) for b in adv.get("fixed", {})), default=None)
        if fixed_v is None and newest_fixed and v[:2] > newest_fixed[:2]:
            continue                       # a branch released after the fix already contains it
        name = f"{adv['id']} ({', '.join(cves)})" if cves else adv["id"]
        if fixed_v and v >= fixed_v:
            continue
        if distro:
            links = tuple(t.format(cve=c) for c in cves for t in TRACKERS)
            out.append(Finding("patch", HIGH, f"Check your distribution's fix for {name}",
                               f"PHP {shown} is a distribution build; distros backport fixes without changing "
                               "the PHP version, so the version alone cannot tell", "Look up these CVEs in your "
                               "distribution's security tracker and confirm the installed package is fixed; "
                               "if not, update the PHP packages now", host, links + refs))
        elif fixed_v:
            out.append(Finding("patch", CRITICAL, f"Vulnerable to {name}",
                               f"PHP {shown} is below the fixed release {fixed} for the {branch} branch. "
                               f"{adv.get('summary', '')}".strip(),
                               f"Update PHP to {fixed} or later now. Until then, put the site behind a web "
                               "application firewall and restrict what is reachable from the internet", host, refs))
        else:
            out.append(Finding("patch", CRITICAL, f"Vulnerable to {name}, no fix for PHP {branch}",
                               f"PHP {shown} is on a branch that receives no fix for this vulnerability",
                               "Upgrade to a supported PHP branch (" +
                               ", ".join(f"{b}: {f}+" for b, f in adv.get("fixed", {}).items()) + "). Until "
                               "then, isolate the server and keep it behind a web application firewall", host, refs))

    branch_info = data["branches"].get(branch)
    if branch_info is None:
        newest = max((parse_version(b) or (0, 0, 0) for b in data["branches"]), default=(0, 0, 0))
        if v[:2] > newest[:2]:
            out.append(Finding("patch", INFO, f"PHP {branch} is newer than the advisory data",
                               "this branch is not in advisories.json yet", "Update advisories.json", host))
        else:
            out.append(Finding("patch", CRITICAL, f"PHP {branch} is long out of support",
                               "this branch has had no security fixes for years", "Upgrade to a supported "
                               "PHP branch", host))
    else:
        until = date.fromisoformat(branch_info["security_until"])
        if until < today and not distro:
            out.append(Finding("patch", HIGH, f"PHP {branch} is end-of-life",
                               f"security support ended {until}; new vulnerabilities will not be fixed upstream",
                               "Plan an upgrade to a supported branch", host))
        elif until < today:
            out.append(Finding("patch", LOW, f"PHP {branch} is end-of-life upstream",
                               f"upstream support ended {until}; your distribution may still patch it",
                               "Confirm your distribution still supports this PHP package", host))
    return out
