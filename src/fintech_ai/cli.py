"""Command-line entry point.

    fintech-ai trend AAPL                 # data + Claude summary
    fintech-ai trend RELIANCE.NS --no-llm # indicators only, no API key needed
    fintech-ai trend MSFT --json          # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence

from fintech_ai.config import load_settings
from fintech_ai.data.indicators import build_snapshot
from fintech_ai.data.market import VALID_PERIODS, fetch_price_history
from fintech_ai.exceptions import ConfigError, FintechAIError
from fintech_ai.llm.client import ClaudeAnalyst

DISCLAIMER = "Research output only - not financial advice. No orders are placed by this tool."

EXIT_OK, EXIT_ERROR, EXIT_CONFIG = 0, 1, 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fintech-ai", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    trend = sub.add_parser("trend", help="Fetch price data and summarize the current trend.")
    trend.add_argument("ticker", help="e.g. AAPL, BRK-B, RELIANCE.NS, ^NSEI")
    trend.add_argument("--period", default="1y", choices=sorted(VALID_PERIODS))
    trend.add_argument("--no-llm", action="store_true", help="Skip the Claude summary.")
    trend.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser


def _run_trend(args: argparse.Namespace) -> int:
    settings = load_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    history = fetch_price_history(args.ticker, period=args.period)
    snapshot = build_snapshot(history)

    analysis = None
    if not args.no_llm:
        analysis = ClaudeAnalyst(settings).summarize_trend(snapshot)

    if args.json:
        out: dict[str, object] = {"snapshot": snapshot, "disclaimer": DISCLAIMER}
        if analysis:
            out["analysis"] = {
                "text": analysis.text,
                "model": analysis.model,
                "truncated": analysis.truncated,
                "usage": {"input": analysis.input_tokens, "output": analysis.output_tokens},
            }
        print(json.dumps(out, indent=2))
        return EXIT_OK

    print(json.dumps(snapshot, indent=2))
    if analysis:
        print("\n" + "=" * 72)
        print(analysis.text)
        if analysis.truncated:
            print("\n[warning] output truncated at max_tokens")
        print("=" * 72)
        print(
            f"model={analysis.model} tokens in/out="
            f"{analysis.input_tokens}/{analysis.output_tokens}"
        )
    print(f"\n{DISCLAIMER}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "trend":
            return _run_trend(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    except FintechAIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        return 130
    return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
