"""Broker gateway.

- zerodha.py: READ-ONLY (holdings, positions, funds, today's orders and trades). No order methods.
- kite_orders.py: Step 7, real orders. Off unless FIN_AGENT_LIVE_TRADING=true. Every order is a draft
  that you approve by typing it; the approval is tied to the SHA-256 of that exact order, works
  once, and expires after 2 minutes; new entries are re-checked by the risk engine when sent.
  Limit and stop-limit orders only. The AI layer has no import path to it (a test checks this).
"""
