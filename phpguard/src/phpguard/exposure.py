"""What does YOUR website reveal to an attacker's first look? Run only against sites you own or are
authorised to test: it refuses to start without that confirmation.

It sends a handful of ordinary GET requests (one per second, a clear User-Agent), never payloads.
"""

import json
import re
import time
import urllib.error
import urllib.request
from typing import Callable
from urllib.parse import urljoin, urlparse

from .findings import CRITICAL, HIGH, INFO, LOW, MEDIUM, Finding

USER_AGENT = "PHPGuard-selfcheck/0.1 (site owner's own security check)"
MAX_BODY = 64 * 1024
Response = tuple[int, dict[str, str], str]           # status, headers (lower-case keys), body (truncated)
Fetch = Callable[[str], Response]


class NotAuthorized(PermissionError):
    pass


def http_fetch(url: str, timeout: int = 10) -> Response:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:     # TLS certificates are verified
            return r.status, {k.lower(): v for k, v in r.headers.items()}, \
                r.read(MAX_BODY).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in (e.headers or {}).items()}, ""


def _env_file(body: str) -> bool:
    return len(re.findall(r"(?m)^[A-Z][A-Z0-9_]{2,}=", body)) >= 2


def _composer(body: str) -> bool:
    try:
        return isinstance(json.loads(body), dict) and ("require" in body or "packages" in body)
    except json.JSONDecodeError:
        return False


# path → (detector on the body, severity, title, fix)
LEAKS: tuple[tuple[str, Callable[[str], bool], str, str, str], ...] = (
    ("/.env", _env_file, CRITICAL, "The .env file (passwords, API keys) is public",
     "Move .env outside the web root or deny it in the web server, then change every secret in it"),
    ("/.git/HEAD", lambda b: b.startswith("ref:") or bool(re.match(r"^[0-9a-f]{40}\s*$", b)), HIGH,
     "The .git folder is public; the whole source code can be downloaded",
     "Deny /.git in the web server (or deploy without it)"),
    ("/phpinfo.php", lambda b: "PHP Version" in b and "phpinfo" in b.lower(), HIGH,
     "A phpinfo() page is public", "Delete it; it lists the exact versions and configuration"),
    ("/info.php", lambda b: "PHP Version" in b and "phpinfo" in b.lower(), HIGH,
     "A phpinfo() page is public", "Delete it; it lists the exact versions and configuration"),
    ("/server-status", lambda b: "Apache Server Status" in b, MEDIUM,
     "Apache server-status is public", "Restrict /server-status to localhost"),
    ("/composer.json", _composer, LOW, "composer.json is public (dependency list)",
     "Keep the project root outside the web root, or deny composer.* files"),
)


def check_site(url: str, authorized: bool, fetch: Fetch = http_fetch, delay: float = 1.0) -> list[Finding]:
    if not authorized:
        raise NotAuthorized("refusing to check a site without --i-own-this-site: only check sites you own "
                            "or have written permission to test")
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("give a full http:// or https:// address")
    host, base = parts.hostname, f"{parts.scheme}://{parts.netloc}/"
    out: list[Finding] = []

    try:
        status, headers, _ = fetch(base)
    except (urllib.error.URLError, OSError, ValueError) as e:
        return [Finding("exposure", INFO, "Site could not be reached", str(e)[:200], "Check the address", host)]

    powered = headers.get("x-powered-by", "")
    if "php" in powered.lower():
        out.append(Finding("hide", MEDIUM, "Response advertises PHP" + (" and its version" if re.search(r"\d", powered)
                                                                        else ""),
                           f"X-Powered-By: {powered}", "Set expose_php = Off in php.ini", host))
    server = headers.get("server", "")
    if re.search(r"/\d", server):
        out.append(Finding("hide", LOW, "Server header reveals the web server version", f"Server: {server}",
                           "nginx: server_tokens off; Apache: ServerTokens Prod and ServerSignature Off", host))
    if parts.scheme == "https" and "strict-transport-security" not in headers:
        out.append(Finding("harden", LOW, "No HSTS header", "browsers may still try plain HTTP first",
                           "Add Strict-Transport-Security: max-age=31536000", host))
    if headers.get("x-content-type-options", "").lower() != "nosniff":
        out.append(Finding("harden", LOW, "No X-Content-Type-Options: nosniff",
                           "browsers may run uploaded files as scripts", "Add X-Content-Type-Options: nosniff", host))
    csp = headers.get("content-security-policy", "")
    if not csp:
        out.append(Finding("harden", LOW, "No Content-Security-Policy", "no browser-side limit on injected scripts",
                           "Add a Content-Security-Policy suited to the site", host))
    if "x-frame-options" not in headers and "frame-ancestors" not in csp:
        out.append(Finding("harden", LOW, "Page can be framed by other sites", "clickjacking is possible",
                           "Add X-Frame-Options: DENY or CSP frame-ancestors 'none'", host))

    for path, detect, sev, title, fix in LEAKS:
        time.sleep(delay)
        try:
            status, _, body = fetch(urljoin(base, path.lstrip("/")))
        except (urllib.error.URLError, OSError, ValueError):
            continue
        if status == 200 and detect(body):
            out.append(Finding("exposure", sev, title, f"{base.rstrip('/')}{path} answered 200", fix, host))
    return out
