"""Closed-loop replay load generator for OpenAI-compatible chat servers.

A workload file holds *units*: a unit is one conversation whose requests are
sent strictly in order (Chat sessions have 4 turns; API and Agentic units have
one request). `concurrency` workers each take the next unit from a shared queue,
so at most `concurrency` requests are in flight and every Chat turn is sent only
after the previous turn of the same session has completed.

Every request is pre-materialised (full message array), so both engines receive
byte-identical requests. Per-request wall-clock timestamps (time.time_ns) are
written to requests.jsonl so that energy can be integrated over exactly the
window between the first dispatch and the last completion.

Usage:
  python -m ww.loadgen --url http://127.0.0.1:8000 --model llama \
      --workload api_small.jsonl --units 40 --out-dir run/measure
  python -m ww.loadgen ... --duration 60      # time-bounded (warm-up)
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import aiohttp

OUTPUT_TOKENS = 128


def build_payload(model: str, messages: list, max_tokens: int) -> dict:
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,          # vLLM + SGLang extension: force exact length
        "ignore_eos": True,                # vLLM + SGLang extension
        "temperature": 0.0,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,         # override model generation_config defaults
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


async def send_one(session: aiohttp.ClientSession, url: str, payload: dict, meta: dict) -> dict:
    rec = dict(meta)
    rec["t_dispatch_ns"] = time.time_ns()
    t_first = None
    t_last = None
    n_chunks = 0
    usage = None
    err = None
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                err = f"HTTP {resp.status}: {(await resp.text())[:300]}"
            else:
                buf = b""
                async for raw in resp.content:
                    buf += raw
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            continue
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if obj.get("usage"):
                            usage = obj["usage"]
                        for ch in obj.get("choices") or []:
                            delta = ch.get("delta") or {}
                            if delta.get("content"):
                                now = time.time_ns()
                                if t_first is None:
                                    t_first = now
                                t_last = now
                                n_chunks += 1
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    rec["t_done_ns"] = time.time_ns()
    rec["t_first_token_ns"] = t_first
    rec["t_last_token_ns"] = t_last
    rec["n_chunks"] = n_chunks
    rec["error"] = err
    if usage:
        rec["prompt_tokens"] = usage.get("prompt_tokens")
        rec["completion_tokens"] = usage.get("completion_tokens")
        # both engines run with cache reporting enabled; SGLang sends null details
        # when nothing was cached, so a missing value means 0 cached tokens
        details = usage.get("prompt_tokens_details") or {}
        rec["cached_tokens"] = details.get("cached_tokens") or 0
    else:
        rec["prompt_tokens"] = rec["completion_tokens"] = rec["cached_tokens"] = None
    return rec


async def run(args) -> dict:
    units = []
    with open(args.workload) as f:
        for line in f:
            if line.strip():
                units.append(json.loads(line))
    if args.units and not args.duration:
        if args.units > len(units):
            raise SystemExit(f"workload has only {len(units)} units, {args.units} requested")
        units = units[: args.units]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    url = args.url.rstrip("/") + "/v1/chat/completions"
    records = []
    deadline = time.time() + args.duration if args.duration else None
    queue: asyncio.Queue = asyncio.Queue()
    if deadline:  # time-bounded: cycle through the pool until the deadline
        idx = 0
        for _ in range(100_000):
            queue.put_nowait(units[idx % len(units)])
            idx += 1
    else:
        for u in units:
            queue.put_nowait(u)

    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency + 4)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async def worker(wid: int):
            while True:
                if deadline and time.time() >= deadline:
                    return
                try:
                    unit = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                for turn, req in enumerate(unit["requests"]):
                    payload = build_payload(args.model, req["messages"], args.max_tokens)
                    meta = {"unit_id": unit["unit_id"], "turn": turn, "worker": wid,
                            "expected_prompt_tokens": req.get("prompt_tokens")}
                    records.append(await send_one(session, url, payload, meta))

        await asyncio.gather(*(worker(i) for i in range(args.concurrency)))

    with open(out_dir / "requests.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    summary = summarize(records)
    summary.update({"workload": str(args.workload), "concurrency": args.concurrency,
                    "units_requested": args.units, "duration_s": args.duration})
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def summarize(records: list) -> dict:
    ok = [r for r in records if not r["error"]]
    s = {"n_requests": len(records), "n_ok": len(ok), "n_errors": len(records) - len(ok)}
    if not ok:
        return s
    t0 = min(r["t_dispatch_ns"] for r in records)
    t1 = max(r["t_done_ns"] for r in records)
    s["t_start_ns"], s["t_end_ns"] = t0, t1
    s["window_s"] = (t1 - t0) / 1e9
    ttft = [(r["t_first_token_ns"] - r["t_dispatch_ns"]) / 1e6 for r in ok if r["t_first_token_ns"]]
    s["ttft_ms_median"] = statistics.median(ttft) if ttft else None
    s["completion_tokens_sum"] = sum(r["completion_tokens"] or 0 for r in ok)
    s["prompt_tokens_sum"] = sum(r["prompt_tokens"] or 0 for r in ok)
    s["req_per_s"] = len(ok) / s["window_s"] if s["window_s"] > 0 else None
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--units", type=int, default=0, help="number of units (0 = all)")
    ap.add_argument("--duration", type=float, default=0, help="time-bounded run in seconds (warm-up)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=OUTPUT_TOKENS)
    ap.add_argument("--request-timeout", type=float, default=1800)
    args = ap.parse_args(argv)
    summary = asyncio.run(run(args))
    print(json.dumps(summary))
    return 0 if summary.get("n_errors", 1) == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
