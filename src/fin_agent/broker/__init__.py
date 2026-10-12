"""Broker gateway.

Now: zerodha.py, a READ-ONLY connection (holdings, positions, funds, today's orders). It has no
method that can place, change or cancel an order.

Invariant for Step 7 (live orders), when it exists:
  - Nothing here can submit an order unless it holds an approval record created by a human
    action (CLI prompt / UI click), tied to the exact payload hash being sent.
  - Paper-trading endpoints only, unless a separate, explicit live-mode flag is set.
  - The LLM layer may only *draft* order payloads; it has no import path to `submit`.
"""
