---
claim_id: 18a33f4c12cea3c004c02fc0
entity: dashboard
status: inferred
confidence: 0.75
sources:
  - dashboard.py:38
  - dashboard.py:688
  - dashboard.py:691
  - dashboard.py:701
  - dashboard.py:755
last_checked: 2026-04-04T19:51:14Z
derived_from:
  - cogops_compiler
  - sast
---

# Module: dashboard

**Language:** py
**Lines of code:** 774

## Types
- `class DashboardHandler(BaseHTTPRequestHandler):` — HTTP handler for the dashboard.

## Functions
- `def record_request(entry: dict)` — Record a proxy request's metrics (called from proxy.py).
- `def log_message(self, format, *args)`
- `def do_GET(self)`
- `def start_dashboard(engine: Any = None, port: int = 9378, daemon: bool = True)` —  Start the dashboard HTTP server in a background thread.  Args: engine: The EntrolyEngine instance to pull real data from. port: Port to serve on (default: 9378). daemon: Run as daemon thread (dies wi
