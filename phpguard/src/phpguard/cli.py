"""phpguard: defend PHP servers you run.

  phpguard collect  [--ini PATH] [--internet-facing] [--criticality high] [-o host.json]   (on each server)
  phpguard audit    host1.json host2.json … [--format text|md|json|csv] [-o report]       (patch + php.ini)
  phpguard check-site https://your-site.example --i-own-this-site                        (what you leak)
  phpguard logs     /var/log/nginx/access.log [--canary /path] [--blocklist ips.txt]      (silent detection)
  phpguard advisories                                                                      (known bugs)
"""

import argparse
import json
import sys
from pathlib import Path

from . import advisories, harden
from .collect import collect
from .exposure import NotAuthorized, check_site
from .findings import Finding
from .logwatch import DEFAULT_CANARIES, RULE_TEXT, scan
from .report import render


def _write(text: str, out: str | None) -> None:
    if out:
        Path(out).write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    else:
        print(text)


def audit_hosts(paths: list[str], data: dict) -> tuple[list[Finding], dict[str, tuple[bool, str]]]:
    findings: list[Finding] = []
    exposure: dict[str, tuple[bool, str]] = {}
    for p in paths:
        try:
            host = json.loads(Path(p).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"cannot read {p}: {e}") from e
        name = str(host.get("host") or Path(p).stem)
        php = host.get("php") or {}
        exposure[name] = (bool(host.get("internet_facing")), str(host.get("criticality", "medium")))
        findings += advisories.assess(php.get("version"), php.get("package_version"), name, data)
        findings += harden.audit({str(k): "" if v is None else str(v) for k, v in (php.get("ini") or {}).items()},
                                 php.get("sapi"), name)
    return findings, exposure


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="phpguard", description="Defend PHP servers you run.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="record this server's PHP version and settings (run on the server)")
    c.add_argument("--php", default="php", help="php binary (default: php)")
    c.add_argument("--ini", action="append", default=[], help="web server's php.ini (repeatable)")
    c.add_argument("--internet-facing", action="store_true")
    c.add_argument("--criticality", choices=("high", "medium", "low"), default="medium")
    c.add_argument("-o", "--out")

    a = sub.add_parser("audit", help="patch status and php.ini hardening for collected hosts")
    a.add_argument("hosts", nargs="+")
    a.add_argument("--advisories", help="custom advisories.json")
    a.add_argument("--format", choices=("text", "md", "json", "csv"), default="text")
    a.add_argument("-o", "--out")

    s = sub.add_parser("check-site", help="what your own site reveals (only sites you own)")
    s.add_argument("url")
    s.add_argument("--i-own-this-site", action="store_true", dest="authorized",
                   help="confirm you own this site or have written permission to test it")
    s.add_argument("--format", choices=("text", "md", "json", "csv"), default="text")

    lg = sub.add_parser("logs", help="find attackers in an access log, silently")
    lg.add_argument("logfile")
    lg.add_argument("--canary", action="append", default=[], help="a path no real visitor opens (repeatable)")
    lg.add_argument("--blocklist", help="write high-risk IPs here (one per line) for your firewall")
    lg.add_argument("--json", action="store_true")

    sub.add_parser("advisories", help="list the known vulnerabilities PHPGuard checks")

    args = ap.parse_args(argv)
    try:
        if args.cmd == "collect":
            _write(json.dumps(collect(args.php, args.ini, args.internet_facing, args.criticality), indent=2),
                   args.out)
        elif args.cmd == "audit":
            findings, exposure = audit_hosts(args.hosts, advisories.load(args.advisories))
            _write(render(findings, args.format, exposure), args.out)
            return 1 if any(f.severity == "critical" for f in findings) else 0     # usable as a CI gate
        elif args.cmd == "check-site":
            print(render(check_site(args.url, args.authorized), args.format))
        elif args.cmd == "logs":
            with open(args.logfile, encoding="utf-8", errors="replace") as f:
                rep = scan(f, tuple(args.canary) or DEFAULT_CANARIES)
            if args.blocklist:
                Path(args.blocklist).write_text("".join(f"{ip}\n" for ip in rep.blocklist()), encoding="utf-8")
            if args.json:
                print(json.dumps({"lines": rep.lines, "unparsed": rep.unparsed,
                                  "attackers": [x.to_dict() for x in rep.attackers]}, indent=2))
            else:
                print(f"{rep.lines} lines read, {rep.unparsed} not in combined format, "
                      f"{len(rep.attackers)} suspicious IPs")
                for x in rep.attackers:
                    flag = "  ← a high-risk request got 200: investigate this first" if x.succeeded else ""
                    print(f"\n[{x.severity.upper()}] {x.ip}  {x.hits} hits  {x.first} … {x.last}{flag}")
                    for r in sorted(x.rules):
                        print(f"   - {RULE_TEXT[r]}")
                    for smp in x.samples:
                        print(f"     e.g. {smp}")
        else:
            data = advisories.load()
            print(f"advisory data updated {data.get('updated')}")
            for adv in data["advisories"]:
                print(f"- {adv['id']} {', '.join(adv.get('cves', []))} [{adv.get('severity')}]: fixed in "
                      + ", ".join(adv.get("fixed", {}).values()))
    except NotAuthorized as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
