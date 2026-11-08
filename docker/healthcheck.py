"""Container health probe. The slim image deliberately has no curl."""
import os
import sys
import urllib.request

port = os.getenv("PORT", "5000")
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=8) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception:
    sys.exit(1)
