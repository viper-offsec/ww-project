"""Smoke test one engine/model: start-up, exact output length, prompt-token
agreement with the generator, cached-token reporting and /metrics families.

  python scripts/smoke_test.py --engine vllm --model llama [--cells api_small,chat_small,agentic_small]
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ww import engines as E  # noqa: E402
from ww import settings as S  # noqa: E402
from ww.metrics_scraper import parse  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=S.ENGINES)
    ap.add_argument("--model", required=True, choices=list(S.MODELS))
    ap.add_argument("--cells", default="api_small,chat_small,agentic_small,api_large")
    ap.add_argument("--units", type=int, default=8)
    args = ap.parse_args(argv)
    out = S.LOGS_DIR / f"smoke_{args.engine}_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    E.kill_leftovers()
    proc = E.start_engine(args.engine, args.model, out / "engine.log")
    report = {"engine": args.engine, "model": args.model, "cells": {}}
    try:
        report["start_s"] = round(E.wait_ready(args.engine, proc), 1)
        print(f"ready after {report['start_s']} s", flush=True)
        for cell in args.cells.split(","):
            profile, context = cell.split("_")
            units = max(1, args.units // (S.CHAT_TURNS if profile == "chat" else 1))
            d = out / cell
            r = subprocess.run([S.runner_python(), "-m", "ww.loadgen", "--url", E.base_url(args.engine),
                                "--model", S.MODELS[args.model]["served_name"],
                                "--workload", str(S.workload_file(args.model, profile, context)),
                                "--units", str(units), "--concurrency", str(S.CONCURRENCY), "--out-dir", str(d)],
                               cwd=S.REPO_ROOT, capture_output=True, text=True)
            recs = [json.loads(x) for x in (d / "requests.jsonl").read_text().splitlines()] if (d / "requests.jsonl").exists() else []
            ok = [x for x in recs if not x["error"]]
            report["cells"][cell] = {
                "rc": r.returncode, "n": len(recs), "errors": sorted({x["error"] for x in recs if x["error"]})[:3],
                "completion_tokens": sorted({x["completion_tokens"] for x in ok}),
                "prompt_minus_expected": sorted({(x["prompt_tokens"] or 0) - (x["expected_prompt_tokens"] or 0) for x in ok}),
                "cached_tokens": [x["cached_tokens"] for x in ok][:8],
                "ttft_ms": [round((x["t_first_token_ns"] - x["t_dispatch_ns"]) / 1e6) for x in ok if x["t_first_token_ns"]][:8],
            }
            print(cell, json.dumps(report["cells"][cell]), flush=True)
        with urllib.request.urlopen(E.base_url(args.engine) + "/metrics", timeout=5) as resp:
            report["metrics"] = parse(resp.read().decode())
        print("metrics:", json.dumps(report["metrics"], indent=1), flush=True)
    finally:
        E.stop_engine(proc, args.engine)
        (out / "smoke_report.json").write_text(json.dumps(report, indent=2))
        print(f"report: {out / 'smoke_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
