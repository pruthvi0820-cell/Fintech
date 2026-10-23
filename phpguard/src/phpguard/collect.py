"""Runs ON a server you administer and records what PHPGuard needs. Read-only, no network.

    phpguard collect --ini /etc/php/8.3/fpm/php.ini --internet-facing --criticality high -o web01.json

The PHP command line (CLI) often uses a different php.ini than the web server (FPM/Apache), so pass the
web server's php.ini with --ini; that is the configuration attackers meet.
"""

import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .harden import parse_ini

_PHP_PROBE = ('echo json_encode(["version" => PHP_VERSION, "sapi" => PHP_SAPI, '
              '"ini" => ini_get_all(null, false), "ini_file" => php_ini_loaded_file()]);')


def _run(cmd: list[str], timeout: int = 15) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def os_release(path: str = "/etc/os-release") -> dict[str, str]:
    out = {}
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                out[k] = v.strip().strip('"')
    except OSError:
        pass
    return {"id": out.get("ID", platform.system().lower()), "version": out.get("VERSION_ID", "")}


def package_version(branch: str | None) -> str | None:
    """The distribution's PHP package version, which shows backported fixes the PHP version hides."""
    if not branch:
        return None
    if shutil.which("dpkg-query"):
        for pkg in (f"php{branch}-common", f"php{branch}-cli", "php-common"):
            if v := _run(["dpkg-query", "-W", "-f=${Version}", pkg]):
                return v
    if shutil.which("rpm"):
        for pkg in ("php-common", "php-cli", "php"):
            if (v := _run(["rpm", "-q", "--qf", "%{VERSION}-%{RELEASE}", pkg])) and "not installed" not in v:
                return v
    if shutil.which("apk"):
        out = _run(["apk", "info", "-v", f"php{branch.replace('.', '')}"])
        if out:
            return out.split()[0]
    return None


def collect(php: str = "php", ini_files: list[str] | None = None, internet_facing: bool = False,
            criticality: str = "medium") -> dict:
    probe = _run([php, "-r", _PHP_PROBE])
    info: dict = {}
    if probe:
        try:
            info = json.loads(probe)
        except json.JSONDecodeError:
            info = {}
    ini = {k: "" if v is None else str(v) for k, v in (info.get("ini") or {}).items()}
    source, sapi = info.get("ini_file") or "php CLI", info.get("sapi")
    for f in ini_files or []:
        try:
            ini.update(parse_ini(Path(f).read_text(encoding="utf-8", errors="replace")))
            source, sapi = f, ("fpm-fcgi" if "fpm" in f else "apache2handler" if "apache" in f else sapi)
        except OSError as e:
            raise ValueError(f"cannot read {f}: {e}") from e
    version = info.get("version")
    branch = ".".join(version.split(".")[:2]) if version else None
    return {
        "host": platform.node() or "unknown",
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": os_release(),
        "internet_facing": internet_facing,
        "criticality": criticality,
        "php": {"version": version, "package_version": package_version(branch), "sapi": sapi,
                "ini_source": source, "ini": ini},
    }
