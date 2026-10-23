# PHPGuard

A defensive toolkit for organisations that run PHP. It answers four questions after a bug like
**PHP2Shell** (CVE-2026-109618 / CVE-2026-109619, patched 21 Oct 2026):

1. **Which of our servers are exposed, and which do we fix first?** (`collect` + `audit`)
2. **Are our PHP settings limiting the damage of the *next* unknown bug?** (`audit`)
3. **What does our website reveal to an attacker's first look?** (`check-site`, own sites only)
4. **Is anyone attacking us right now?** (`logs`, with canary traps and a blocklist)

It only reads systems you control. It sends no exploit payloads and makes no changes; you apply the
fixes it recommends. Standard library only, so it runs on any server with Python 3.10+.

## The idea: defend like an attacker hides

| Attackers… | PHPGuard helps you… |
|---|---|
| pick targets by reading version banners and error pages | **hide**: `expose_php = Off`, `display_errors = Off`, no version in `Server`, no `phpinfo.php` |
| exploit bugs that already have a patch | **patch fast**: version vs advisory file, distro backports handled, end-of-life branches flagged |
| use one bug to take the whole server | **limit damage**: `disable_functions`, `open_basedir`, `allow_url_include = Off`, safe sessions |
| probe quietly before attacking | **watch silently**: log rules for webshells, PHP wrappers, PHP-CGI argument injection, traversal, recon |
| explore everything they find | **lay traps**: canary paths no real visitor opens; any hit is an attacker, with almost no false alarms |

## Use it

```bash
cd phpguard && pip install -e .

# 1. On each server (read-only; pass the WEB server's php.ini, the CLI one often differs)
phpguard collect --ini /etc/php/8.3/fpm/php.ini --internet-facing --criticality high -o web01.json

# 2. Centrally: patch status + hardening, graded A–F, in fix-first order
phpguard audit web01.json web02.json legacy.json            # text
phpguard audit *.json --format csv -o findings.csv          # for Excel / Power BI
# exit code 1 when anything critical is open, so it can fail a CI/CD or compliance pipeline

# 3. Your own site's first impression (refuses to run without the confirmation flag)
phpguard check-site https://www.your-company.example --i-own-this-site

# 4. Silent detection, e.g. every 10 minutes from cron
phpguard logs /var/log/nginx/access.log --canary /db_backup.sql --canary /old-admin/ --blocklist /etc/phpguard/block.txt
```

Try it on the bundled sample company (4 servers, one attacked access log). [DEMO.md](DEMO.md) shows the real output:

```bash
phpguard audit samples/*.json
phpguard logs samples/access.log --canary /db_backup.sql
```

## Rolling it out in a company

- **Collect** on every PHP host with your config-management tool (Ansible, which the FinTray repo already
  uses) or a daily cron job; gather the JSON files centrally.
- **Audit** daily; send the CSV to the dashboard and the critical items to the on-call channel.
- **Block**: feed `block.txt` to the firewall (nginx `deny`, fail2ban, or the cloud WAF's IP list). Review
  before blocking automatically: shared corporate IPs can appear in logs.
- **Canaries**: pick 3–5 paths that look valuable but do not exist (`/db_backup.sql`, `/old-admin/`), never
  link to them, and alert on any hit.
- **New bug?** Add one entry to `src/phpguard/advisories.json` (fixed version per branch). No code changes.

## Limits (read before relying on it)

- **Version checks are only as current as `advisories.json`.** The tool warns when the file is over 30 days old.
- **Distribution packages** (Ubuntu, Debian, Red Hat) are flagged "check your distribution's fix", never
  declared safe or vulnerable from the version number, because distros backport fixes.
- **Log rules are signatures of well-known attack patterns.** They will miss new techniques and do not
  detect PHP2Shell's specific exploit, whose details are not published here. Canaries and the
  "high-risk request got 200" flag are the strongest signals.
- **It does not block anything itself** and is not a WAF. Put internet-facing PHP behind one; during
  PHP2Shell, WAF providers blocked attacks before the patch was out.
- **`check-site`** sends a handful of ordinary GET requests, one per second. Use it only on sites you own or
  have written permission to test.

## Layout

```
src/phpguard/
  advisories.py / advisories.json   known bugs, fixed versions, end-of-life dates, distro backport handling
  collect.py                        on-server, read-only inventory (version, package, php.ini)
  harden.py                         php.ini baseline (hide, limit damage, safe sessions)
  exposure.py                       owner-only website self-check (banners, headers, leaked files)
  logwatch.py                       access-log rules, canaries, scanner detection, blocklist
  report.py                         A–F grades, fix-first ordering, text / Markdown / JSON / CSV
  cli.py                            the `phpguard` command
```

`python -m pytest -q && python -m ruff check src tests`
