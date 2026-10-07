"""Phase 3: broker gateway. Intentionally empty until then.

Invariant this package will enforce when it exists:
  - Nothing here can submit an order unless it holds an approval record created by a human
    action (CLI prompt / UI click), tied to the exact payload hash being sent.
  - Paper-trading endpoints only, unless a separate, explicit live-mode flag is set.
  - The LLM layer may only *draft* order payloads; it has no import path to `submit`.
"""
