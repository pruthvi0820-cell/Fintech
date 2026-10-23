"""php.ini audit, log detection and the owner-only site check."""

import pytest

from phpguard.exposure import NotAuthorized, check_site
from phpguard.harden import audit, parse_ini, size_mb
from phpguard.logwatch import scan

WEAK_INI = {"expose_php": "1", "display_errors": "On", "log_errors": "0", "allow_url_include": "1",
            "allow_url_fopen": "1", "disable_functions": "exec", "open_basedir": "", "session.cookie_httponly": "",
            "session.cookie_secure": "0", "session.use_strict_mode": "0", "session.cookie_samesite": "",
            "cgi.fix_pathinfo": "1", "enable_dl": "0", "file_uploads": "1", "upload_max_filesize": "128M"}
STRONG_INI = {"expose_php": "Off", "display_errors": "Off", "log_errors": "On", "allow_url_include": "Off",
              "allow_url_fopen": "Off", "disable_functions": "exec,shell_exec,system,passthru,proc_open,popen",
              "open_basedir": "/var/www/site:/tmp", "session.cookie_httponly": "1", "session.cookie_secure": "1",
              "session.use_strict_mode": "1", "session.cookie_samesite": "Lax", "cgi.fix_pathinfo": "0",
              "file_uploads": "On", "upload_max_filesize": "8M"}


def test_weak_config_flags_everything():
    titles = {f.title for f in audit(WEAK_INI, "fpm-fcgi", "web01")}
    assert "Server advertises its PHP version" in titles and "Errors are shown to visitors" in titles
    assert "Remote code inclusion is allowed" in titles and "cgi.fix_pathinfo is on under PHP-FPM" in titles
    assert len(titles) == 13


def test_strong_config_is_clean_and_missing_keys_are_not_assumed():
    assert audit(STRONG_INI, "fpm-fcgi") == []
    assert audit({}, "fpm-fcgi") == []


def test_ini_parsing_and_sizes():
    ini = parse_ini("[PHP]\n; comment\nexpose_php = Off ; trailing\nerror_log = \"/var/log/php.log\"\n")
    assert ini == {"expose_php": "Off", "error_log": "/var/log/php.log"}
    assert (size_mb("2M"), size_mb("1G"), size_mb("512K"), size_mb("bad")) == (2, 1024, 0.5, 0)


LOG = [
    '203.0.113.5 - - [23/Oct/2026:10:00:01 +0530] "GET /wp-content/uploads/2026/x.php?c=id HTTP/1.1" 200 12 "-" "-"',
    '203.0.113.5 - - [23/Oct/2026:10:00:02 +0530] "GET /index.php?page=php://filter/resource=index HTTP/1.1" 200 9 "-" "-"',
    '198.51.100.7 - - [23/Oct/2026:10:01:00 +0530] "GET /.env HTTP/1.1" 404 0 "-" "-"',
    '198.51.100.7 - - [23/Oct/2026:10:01:01 +0530] "GET /db_backup.sql HTTP/1.1" 404 0 "-" "-"',
    '192.0.2.9 - - [23/Oct/2026:10:02:00 +0530] "POST /php-cgi/php-cgi.exe?%ADd+allow_url_include%3D1 HTTP/1.1" 404 0 "-" "-"',
    '192.0.2.10 - - [23/Oct/2026:10:03:00 +0530] "GET /download.php?f=..%2f..%2fetc%2fpasswd HTTP/1.1" 403 0 "-" "-"',
    '10.0.0.2 - - [23/Oct/2026:10:04:00 +0530] "GET /about.php HTTP/1.1" 200 5120 "-" "Mozilla/5.0"',
    'not a log line',
]


def test_log_scan_finds_attackers_and_ranks_successes_first():
    rep = scan(LOG, canaries=("/db_backup.sql",))
    by_ip = {a.ip: a for a in rep.attackers}
    assert rep.lines == 8 and rep.unparsed == 1 and "10.0.0.2" not in by_ip      # normal visitor ignored
    first = rep.attackers[0]
    assert first.ip == "203.0.113.5" and first.succeeded and first.severity == "critical"
    assert first.rules == {"webshell", "injection"}
    assert by_ip["198.51.100.7"].rules == {"recon", "canary"} and by_ip["198.51.100.7"].severity == "critical"
    assert by_ip["192.0.2.9"].rules == {"php-cgi"} and by_ip["192.0.2.10"].rules == {"traversal"}
    assert set(rep.blocklist()) == {"203.0.113.5", "198.51.100.7", "192.0.2.9", "192.0.2.10"}


def test_scanner_detected_by_404_burst():
    lines = [f'198.51.100.99 - - [23/Oct/2026:11:00:{i:02d} +0530] "GET /page{i} HTTP/1.1" 404 0 "-" "-"'
             for i in range(35)]
    rep = scan(lines)
    assert rep.attackers[0].rules == {"scanner"} and rep.attackers[0].severity == "medium"


def fake_site(pages):
    calls = []

    def fetch(url):
        calls.append(url)
        return pages.get(url, (404, {}, ""))
    return fetch, calls


def test_site_check_requires_ownership():
    with pytest.raises(NotAuthorized):
        check_site("https://example.com", authorized=False)
    with pytest.raises(ValueError):
        check_site("ftp://example.com", authorized=True)


def test_site_check_reports_leaks_and_hidden_headers():
    fetch, calls = fake_site({
        "https://shop.example/": (200, {"x-powered-by": "PHP/8.1.2", "server": "nginx/1.18.0"}, "<html>"),
        "https://shop.example/.env": (200, {}, "APP_KEY=abc\nDB_PASSWORD=secret\n"),
        "https://shop.example/.git/HEAD": (200, {}, "ref: refs/heads/main\n"),
        "https://shop.example/phpinfo.php": (200, {}, "<title>phpinfo()</title> PHP Version 8.1.2"),
    })
    fs = check_site("https://shop.example/some/page", authorized=True, fetch=fetch, delay=0)
    titles = {f.title: f.severity for f in fs}
    assert titles["The .env file (passwords, API keys) is public"] == "critical"
    assert titles["The .git folder is public; the whole source code can be downloaded"] == "high"
    assert titles["A phpinfo() page is public"] == "high"
    assert titles["Response advertises PHP and its version"] == "medium"
    assert "Server header reveals the web server version" in titles and "No HSTS header" in titles
    assert all(c.startswith("https://shop.example/") for c in calls) and len(calls) == 7


def test_site_check_quiet_site_and_unreachable():
    fetch, _ = fake_site({"https://ok.example/": (200, {
        "strict-transport-security": "max-age=31536000", "x-content-type-options": "nosniff",
        "content-security-policy": "default-src 'self'; frame-ancestors 'none'"}, "<html>"),
        "https://ok.example/.env": (200, {}, "<html>not found page</html>")})       # soft 404 is not a leak
    assert check_site("https://ok.example", authorized=True, fetch=fetch, delay=0) == []

    def down(url):
        raise OSError("connection refused")
    assert check_site("https://down.example", authorized=True, fetch=down, delay=0)[0].severity == "info"
