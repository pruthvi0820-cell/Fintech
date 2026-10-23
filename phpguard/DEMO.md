# PHPGuard demo output

Real output from the bundled sample data (`samples/`). Regenerate with the commands shown.

## Known bugs checked

```
$ phpguard advisories
advisory data updated 2026-10-23
- PHP2Shell CVE-2026-109618, CVE-2026-109619 [critical]: fixed in 8.5.12, 8.4.27, 8.3.36, 8.2.35
```

## Fleet audit: 4 sample servers

```
$ phpguard audit samples/*.json
PHPGuard report

api-web02: grade A (100/100)
intranet-01: grade B (84/100)
legacy-erp: grade F (1/100)
shop-web01: grade F (11/100)

Fix in this order:
  1. [CRITICAL] shop-web01: Vulnerable to PHP2Shell (CVE-2026-109618, CVE-2026-109619)
     PHP 8.3.30 is below the fixed release 8.3.36 for the 8.3 branch. Unauthenticated remote code execution in the PHP runtime, affecting PHP installations that serve web requests.
     Fix: Update PHP to 8.3.36 or later now. Until then, put the site behind a web application firewall and restrict what is reachable from the internet
     See: https://www.cisa.gov/news-events/alerts/2026/10/21/critical-vulnerabilities-php-require-immediate-patching
     See: https://cert.europa.eu/publications/security-advisories/2026-018
  2. [CRITICAL] legacy-erp: Vulnerable to PHP2Shell (CVE-2026-109618, CVE-2026-109619), no fix for PHP 7.4
     PHP 7.4.33 is on a branch that receives no fix for this vulnerability
     Fix: Upgrade to a supported PHP branch (8.5: 8.5.12+, 8.4: 8.4.27+, 8.3: 8.3.36+, 8.2: 8.2.35+). Until then, isolate the server and keep it behind a web application firewall
     See: https://www.cisa.gov/news-events/alerts/2026/10/21/critical-vulnerabilities-php-require-immediate-patching
     See: https://cert.europa.eu/publications/security-advisories/2026-018
  3. [HIGH    ] shop-web01: Errors are shown to visitors
     error messages reveal file paths, queries and code structure
     Fix: Set display_errors = Off (keep log_errors = On)
  4. [HIGH    ] legacy-erp: Errors are shown to visitors
     error messages reveal file paths, queries and code structure
     Fix: Set display_errors = Off (keep log_errors = On)
  5. [HIGH    ] legacy-erp: PHP 7.4 is end-of-life
     security support ended 2022-11-28; new vulnerabilities will not be fixed upstream
... (26 findings in total)
```

## Attack detection in an access log

```
$ phpguard logs samples/access.log --canary /db_backup.sql --blocklist block.txt
10 lines read, 0 not in combined format, 4 suspicious IPs

[CRITICAL] 203.0.113.5  2 hits  23/Oct/2026:09:20:11 +0530 … 23/Oct/2026:09:20:14 +0530  ← a high-risk request got 200: investigate this first
   - PHP file requested inside an upload or media folder (possible webshell)
     e.g. POST /wp-content/uploads/2026/10/cache.php → 200
     e.g. GET /wp-content/uploads/2026/10/cache.php?c=whoami → 200

[CRITICAL] 198.51.100.7  3 hits  23/Oct/2026:09:31:00 +0530 … 23/Oct/2026:09:31:02 +0530
   - Opened a canary path that no real visitor uses
   - Probing for secret or admin files
     e.g. GET /.env → 404
     e.g. GET /.git/config → 404
     e.g. GET /db_backup.sql → 404

[HIGH] 192.0.2.10  1 hits  23/Oct/2026:09:41:00 +0530 … 23/Oct/2026:09:41:00 +0530
   - Path traversal attempt (../)
     e.g. GET /download.php?f=..%2f..%2f..%2fetc%2fpasswd → 403

[HIGH] 192.0.2.9  1 hits  23/Oct/2026:09:40:00 +0530 … 23/Oct/2026:09:40:00 +0530
   - Code injection attempt (PHP wrapper, eval or shell function in the request)
     e.g. POST /index.php?page=php://input → 403

$ cat block.txt
203.0.113.5
198.51.100.7
192.0.2.10
192.0.2.9
```

## Website self-check refuses without the owner's confirmation

```
$ phpguard check-site https://example.com
error: refusing to check a site without --i-own-this-site: only check sites you own or have written permission to test
```
