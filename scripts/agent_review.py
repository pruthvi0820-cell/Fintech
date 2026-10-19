"""Trade-review agent in the terminal: plan -> risk rules -> library lookup -> AI review -> you approve.

Usage:
    python scripts/agent_review.py TCS.NS --reason "Breakout above resistance"
    python scripts/agent_review.py TCS.NS --semantic        # meaning-based library search (Ollama embeddings)
    python scripts/agent_review.py TCS.NS --no-llm          # code facts only, no model

Paper money only. Needs: pip install -e ".[agent]"
"""

from __future__ import annotations

import argparse
import sys
import uuid

from fin_agent.agents import trade_review as agent
from fin_agent.knowledge import library as library_mod
from fin_agent.knowledge import vectors
from fin_agent.portfolio import paper as paper_mod


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Propose a paper trade; open it only if you approve.")
    p.add_argument("ticker")
    p.add_argument("--reason", default="", help="why you are considering the trade")
    p.add_argument("--semantic", action="store_true", help="add meaning-based library search")
    p.add_argument("--no-llm", action="store_true", help="skip the AI review")
    args = p.parse_args()

    client = None
    if not args.no_llm:
        from fin_agent.llm.factory import client_from_env
        client = client_from_env()
    lib = library_mod.Library(library_mod.default_path())
    index = None
    if args.semantic:
        try:
            index = vectors.VectorIndex(vectors.default_vector_path(lib), vectors.embedder_from_env())
            print(f"[vectors] {index.sync(lib)}")
        except vectors.VectorError as exc:
            print(f"[vectors] {exc} Using keyword search.", file=sys.stderr)
            index = None
    try:
        graph = agent.build_graph(agent.AgentTools(client=client, book=paper_mod.PaperBook(paper_mod.default_path()),
                                                   library=lib, index=index))
        thread = uuid.uuid4().hex
        step = agent.start_review(graph, thread, args.ticker, args.reason)
    except agent.AgentError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    s = step.state
    if plan := s.get("plan"):
        print(f"\nPlan: {plan['shares']} x {args.ticker.upper()} entry ₹{plan['entry']:,.2f} stop ₹{plan['stop']:,.2f} "
              f"target ₹{plan['target']:,.2f} max loss ₹{plan['max_loss']:,.2f}")
    for item in (s.get("checklist") or {}).get("items", []):
        print(f"  [{item['status']:>7}] {item['name']}: {item['detail']}")
    if retrieval := s.get("retrieval"):
        print(f"\nLibrary search: {retrieval['mode']}" + (f" ({retrieval['note']})" if retrieval.get("note") else ""))
    if s.get("review"):
        print("\n" + s["review"] + ("\n\n" + s["review_check"] if s.get("review_check") else ""))
    if step.pending is not None:
        yes = input("\nOpen this paper trade? [y/N] ").strip().lower() == "y"
        note = input("Note (optional): ").strip()
        s = agent.resume_review(graph, thread, approved=yes, note=note).state
    outcome = s.get("outcome") or {}
    print(f"\n[{outcome.get('status', '?')}] {outcome.get('message', '')}")
    return 0 if outcome.get("status") in ("opened", "rejected") else 1


if __name__ == "__main__":
    sys.exit(main())
