"""Manage the trusted library from the command line (handy for big PDFs), and test its search
without the AI model.

Usage:
    python scripts/library.py --list
    python scripts/library.py --add "D:\\docs\\budget_speech_2024.pdf" --title "Budget speech 2024-25" \
        --kind official --publisher "Ministry of Finance" --published 2024-07-23
    python scripts/library.py --search "long term capital gains exemption"
    python scripts/library.py --remove 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fin_agent.knowledge.library import KINDS, Library, LibraryError, default_path


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="FinTray trusted library.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="list the documents")
    g.add_argument("--add", metavar="FILE", help="add a PDF, TXT or MD file")
    g.add_argument("--search", metavar="QUESTION", help="show the passages a question would be given")
    g.add_argument("--remove", type=int, metavar="ID", help="remove document lib #ID from search")
    p.add_argument("--title", default="")
    p.add_argument("--kind", choices=list(KINDS), default="other")
    p.add_argument("--publisher")
    p.add_argument("--published", help="YYYY-MM-DD")
    p.add_argument("--url")
    args = p.parse_args()

    lib = Library(default_path())
    try:
        if args.add:
            path = Path(args.add)
            doc = lib.add(path.name, path.read_bytes(), args.title, args.kind, args.publisher, args.published,
                          args.url)
            print(f"Added lib #{doc.id}: {doc.describe()}, {doc.passages} passages.")
        elif args.remove is not None:
            doc = lib.remove(args.remove)
            print(f"Removed lib #{doc.id} ({doc.title}) from search.")
        elif args.search:
            found = lib.search(args.search)
            if not found:
                print("No passage shares at least 2 meaningful words with that question.")
            for p_ in found:
                print(f"--- [{p_.citation()}] score {p_.score} · {p_.doc.describe()}")
                print(p_.text[:600] + ("…" if len(p_.text) > 600 else ""))
        else:
            docs = lib.documents()
            if not docs:
                print(f"The library is empty ({default_path()}).")
            for d in docs:
                print(f"lib #{d.id:<3} {d.describe()} · {d.passages} passages · {d.file_name}")
    except (LibraryError, OSError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
