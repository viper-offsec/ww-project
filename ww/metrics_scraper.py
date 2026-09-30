"""Scrape an engine's Prometheus /metrics endpoint periodically (until SIGTERM).

Only metric families relevant to the study are kept (KV-cache usage, prefix
cache, preemptions, queue/running requests, token counters). Values of the same
family are summed over label sets. Output: one JSON object per line
{"t_ns": ..., "m": {"vllm:kv_cache_usage_perc": 0.12, ...}}.

Usage: python -m ww.metrics_scraper --url http://127.0.0.1:8000 --out metrics.jsonl
"""
import argparse
import json
import re
import signal
import sys
import time
import urllib.request

KEEP = re.compile(r"(kv_cache_usage|cache_usage_perc|token_usage|cache_hit|prefix_cache|cached_tokens|"
                  r"preempt|retract|num_requests_running|num_requests_waiting|num_running_reqs|"
                  r"num_queue_reqs|prompt_tokens_total|generation_tokens_total|num_used_tokens)")
LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)")

_running = True


def _stop(*_):
    global _running
    _running = False


def parse(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = LINE.match(line)
        if not m or not KEEP.search(m.group(1)):
            continue
        name = m.group(1)
        if name.endswith(("_bucket", "_created")):
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=0.5)
    args = ap.parse_args(argv)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    url = args.url.rstrip("/") + "/metrics"
    with open(args.out, "w") as f:
        while _running:
            t = time.time_ns()
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    m = parse(r.read().decode("utf-8", "replace"))
                f.write(json.dumps({"t_ns": t, "m": m}) + "\n")
            except Exception as e:  # noqa: BLE001
                f.write(json.dumps({"t_ns": t, "error": str(e)[:200]}) + "\n")
            f.flush()
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
