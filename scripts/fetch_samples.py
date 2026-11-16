"""Download the demo PDFs used in the README and smoke test.

Usage:
    python scripts/fetch_samples.py

The 15-page paper is not committed to keep the repository small; this script
fetches it from arXiv on demand.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"

FILES = {
    "attention-is-all-you-need.pdf": "https://arxiv.org/pdf/1706.03762",
    "sample-1page.pdf": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
}


def main() -> int:
    SAMPLES.mkdir(exist_ok=True)
    for name, url in FILES.items():
        target = SAMPLES / name
        if target.exists():
            print(f"skip  {name} (already present)")
            continue
        print(f"fetch {name} <- {url}")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "pdf-summarizer-ai/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                target.write_bytes(response.read())
        except Exception as exc:  # noqa: BLE001
            print(f"  failed: {exc}")
            return 1
        print(f"  saved {target.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
