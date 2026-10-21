"""python -m ats_analyzer resume.pdf  →  the parsed profile and findings as JSON."""

import argparse
import json
import sys

from .extract import ExtractionError
from .parser import parse_file


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ats-parse", description="Show what an ATS reads from a resume.")
    ap.add_argument("resume", help="PDF, DOCX or UTF-8 text file")
    ap.add_argument("--findings-only", action="store_true", help="print only the problems found")
    args = ap.parse_args(argv)
    try:
        parsed = parse_file(args.resume)
    except ExtractionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    data = parsed.to_dict()
    if args.findings_only:
        data = {"findings": data["findings"], "experience_months": data["experience_months"]}
    json.dump(data, sys.stdout, indent=2, ensure_ascii=False)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
