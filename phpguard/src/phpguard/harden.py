"""php.ini settings that limit the damage of the next unknown bug, and keep the server quiet about itself.

Only settings present in the collected configuration are judged; a missing key is never assumed bad.
"""

import re

from .findings import HIGH, LOW, MEDIUM, Finding

DANGEROUS_FUNCTIONS = ("exec", "shell_exec", "system", "passthru", "proc_open", "popen")
_ON = {"1", "on", "true", "yes"}
_SIZE = re.compile(r"^\s*(\d+)\s*([kmg]?)", re.I)


def on(value: str | None) -> bool:
    return (value or "").strip().lower() in _ON


def size_mb(value: str | None) -> float:
    m = _SIZE.match(value or "")
    if not m:
        return 0.0
    n, unit = int(m.group(1)), m.group(2).lower()
    return n / 1024 / 1024 if not unit else {"k": n / 1024, "m": float(n), "g": n * 1024.0}[unit]


def parse_ini(text: str) -> dict[str, str]:
    """Minimal php.ini reader: `key = value` lines, ';' comments, quotes stripped, [sections] ignored."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line or line.startswith("[") or "=" not in line:
            continue
        key, value = (p.strip() for p in line.split("=", 1))
        out[key] = value.strip("\"'")
    return out


def audit(ini: dict[str, str], sapi: str | None = None, host: str = "") -> list[Finding]:
    out: list[Finding] = []

    def rule(key: str, bad: bool, area: str, sev: str, title: str, detail: str, fix: str) -> None:
        if key in ini and bad:
            out.append(Finding(area, sev, title, detail, fix, host))

    rule("expose_php", on(ini.get("expose_php")), "hide", MEDIUM, "Server advertises its PHP version",
         "every response carries an X-Powered-By: PHP/x.y.z header, which tells attackers exactly which "
         "exploits to try", "Set expose_php = Off")
    rule("display_errors", on(ini.get("display_errors")) and (sapi or "") != "cli", "hide", HIGH,
         "Errors are shown to visitors", "error messages reveal file paths, queries and code structure",
         "Set display_errors = Off (keep log_errors = On)")
    rule("log_errors", not on(ini.get("log_errors")), "detect", MEDIUM, "Errors are not logged",
         "attack attempts that cause errors leave no trace", "Set log_errors = On and point error_log at a "
         "file you monitor")
    rule("allow_url_include", on(ini.get("allow_url_include")), "harden", HIGH, "Remote code inclusion is allowed",
         "include() can load code from a URL, the classic remote file inclusion hole", "Set allow_url_include = Off")
    rule("allow_url_fopen", on(ini.get("allow_url_fopen")), "harden", LOW, "PHP can open remote URLs as files",
         "makes server-side request forgery and remote payloads easier", "Set allow_url_fopen = Off unless the "
         "application needs it")
    if "disable_functions" in ini:
        disabled = {f.strip().lower() for f in ini["disable_functions"].split(",") if f.strip()}
        missing = [f for f in DANGEROUS_FUNCTIONS if f not in disabled]
        if missing:
            out.append(Finding("harden", MEDIUM, "Shell-command functions are enabled",
                               f"an attacker who gets code running can call {', '.join(missing)} to control the "
                               "server", "Add them to disable_functions (test the application first; some need "
                               "one of them)", host))
    rule("open_basedir", not (ini.get("open_basedir") or "").strip(), "harden", MEDIUM,
         "PHP can read any file on the server", "without open_basedir, a bug in one site can read other sites' "
         "files and system files", "Set open_basedir to the site's own directories")
    rule("session.cookie_httponly", not on(ini.get("session.cookie_httponly")), "harden", MEDIUM,
         "Session cookies are readable by scripts", "a cross-site scripting bug can steal logged-in sessions",
         "Set session.cookie_httponly = 1")
    rule("session.cookie_secure", not on(ini.get("session.cookie_secure")), "harden", MEDIUM,
         "Session cookies can travel over plain HTTP", "sessions can be intercepted on public networks",
         "Set session.cookie_secure = 1 (site must use HTTPS)")
    rule("session.use_strict_mode", not on(ini.get("session.use_strict_mode")), "harden", LOW,
         "Session fixation is possible", "PHP accepts session IDs it never issued", "Set session.use_strict_mode = 1")
    rule("session.cookie_samesite", not (ini.get("session.cookie_samesite") or "").strip(), "harden", LOW,
         "Session cookies have no SameSite policy", "weaker protection against cross-site request forgery",
         "Set session.cookie_samesite = Lax (or Strict)")
    rule("cgi.fix_pathinfo", on(ini.get("cgi.fix_pathinfo")) and (sapi or "").startswith("fpm"), "harden", MEDIUM,
         "cgi.fix_pathinfo is on under PHP-FPM", "with some web-server setups, an uploaded image can be run as "
         "PHP code", "Set cgi.fix_pathinfo = 0")
    rule("enable_dl", on(ini.get("enable_dl")), "harden", LOW, "Scripts can load PHP extensions at runtime",
         "code could load native extensions", "Set enable_dl = Off")
    rule("file_uploads", on(ini.get("file_uploads")) and size_mb(ini.get("upload_max_filesize")) > 20, "harden",
         LOW, "Large uploads are accepted", f"upload_max_filesize is {ini.get('upload_max_filesize')}",
         "Lower upload_max_filesize to what the site needs")
    return out
