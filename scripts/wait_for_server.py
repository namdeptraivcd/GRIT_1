#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(args.url, timeout=3) as response:
                if 200 <= response.status < 300:
                    print(f"ready: {args.url}", flush=True)
                    return
        except Exception as error:
            last_error = error
        time.sleep(2)
    raise SystemExit(f"server not ready after {args.timeout}s: {args.url}; last_error={last_error}")


if __name__ == "__main__":
    main()
