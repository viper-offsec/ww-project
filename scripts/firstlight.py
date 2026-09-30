"""Minimal engine check that needs no workloads: start the engine with the
experiment flags, send the same request twice (exact 128 output tokens?
cached tokens reported on the repeat?), dump relevant /metrics, stop.

  python scripts/firstlight.py --engine vllm --model llama
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ww import engines as E  # noqa: E402
from ww import settings as S  # noqa: E402
from ww.loadgen import build_payload  # noqa: E402
from ww.metrics_scraper import parse  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=S.ENGINES)
    ap.add_argument("--model", default="llama", choices=list(S.MODELS))
    args = ap.parse_args(argv)
    log = S.LOGS_DIR / f"firstlight_{args.engine}_{args.model}.log"
    E.kill_leftovers()
    p = E.start_engine(args.engine, args.model, log)
    try:
        print("READY_S", round(E.wait_ready(args.engine, p), 1), flush=True)
        msgs = [{"role": "system", "content": S.API_SYSTEM_PROMPT},
                {"role": "user", "content": "Explain what a KV cache is in two sentences. " * 20}]
        for i in range(2):
            pl = build_payload(S.MODELS[args.model]["served_name"], msgs, S.OUTPUT_TOKENS)
            pl["stream"] = False
            pl.pop("stream_options")
            req = urllib.request.Request(E.base_url(args.engine) + "/v1/chat/completions",
                                         data=json.dumps(pl).encode(), headers={"Content-Type": "application/json"})
            t0 = time.time()
            r = json.loads(urllib.request.urlopen(req, timeout=900).read())
            print(f"REQ {i} {time.time() - t0:.2f}s usage={json.dumps(r.get('usage'))}", flush=True)
        m = parse(urllib.request.urlopen(E.base_url(args.engine) + "/metrics", timeout=10).read().decode())
        print("METRICS", json.dumps(m, indent=1))
    finally:
        E.stop_engine(p, args.engine)
    return 0


if __name__ == "__main__":
    sys.exit(main())
