"""Generate the fixed request sets of every (model, profile, context) cell.

Token budgets are exact with respect to each model's own tokenizer *including
its chat template* (the same count the engines report as prompt_tokens).

  API     : fixed system prompt + unique user prompt; prompt = C tokens
  Chat    : 4-turn sessions; turn t resends the full history (previous user
            messages + canned 128-token assistant replies) + a 64-token user
            message; turn 1 is sized so the mean prompt over the 4 turns is C
  Agentic : one global system prefix with tool schemas (0.75*C tokens, shared
            by all requests of the cell) + a unique task/observation part

Sources (raw corpora in ~/greenlab_exp/load_raw):
  ShareGPT V3 (anon8231489123/ShareGPT_Vicuna_unfiltered) - user/assistant texts
  Glaive function calling v2 (glaiveai/glaive-function-calling-v2) - tool schemas, tasks

Outputs: workloads/<model>/<profile>_<context>[_warmup].jsonl + manifest.json
"""
import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ww import settings as S  # noqa: E402


# ------------------------------------------------------------------ corpora
def mostly_ascii(t: str) -> bool:
    return sum(c.isascii() for c in t) / max(1, len(t)) > 0.95


def load_sharegpt(path: Path, limit: int = 40000):
    data = json.loads(path.read_text())
    human, gpt = [], []
    for conv in data:
        for turn in conv.get("conversations", []):
            v = (turn.get("value") or "").strip()
            if len(v) < 80 or not mostly_ascii(v):
                continue
            (human if turn.get("from") == "human" else gpt).append(v)
        if len(human) > limit and len(gpt) > limit:
            break
    return human, gpt


def load_glaive(path: Path, limit: int = 20000):
    data = json.loads(path.read_text())
    tools, tasks = [], []
    for row in data:
        sysmsg = row.get("system", "")
        if "{" in sysmsg:
            schema = sysmsg[sysmsg.find("{"):].strip()
            if 40 < len(schema) < 6000:
                tools.append(schema)
        m = re.search(r"USER:(.*?)(?:ASSISTANT:|$)", row.get("chat", ""), re.S)
        if m and len(m.group(1).strip()) > 20:
            tasks.append(m.group(1).strip())
        if len(tools) > limit and len(tasks) > limit:
            break
    return list(dict.fromkeys(tools)), tasks


# ------------------------------------------------------------------ token helpers
class Tok:
    def __init__(self, model_path: Path):
        from transformers import AutoTokenizer
        self.t = AutoTokenizer.from_pretrained(str(model_path))

    def ids(self, text: str) -> list:
        return self.t(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids: list) -> str:
        return self.t.decode(ids, skip_special_tokens=True)

    def prompt_ids(self, messages: list, gen: bool = True) -> list:
        text = self.t.apply_chat_template(messages, tokenize=False, add_generation_prompt=gen)
        return self.ids(text)

    def prompt_len(self, messages: list) -> int:
        return len(self.prompt_ids(messages))


class Source:
    """Draws random texts from a corpus and cuts them to token budgets."""
    def __init__(self, texts, tok: Tok, rng: random.Random):
        self.texts, self.tok, self.rng = texts, tok, rng

    def base_ids(self, n: int) -> list:
        ids = []
        while len(ids) < n:
            ids += self.tok.ids(self.rng.choice(self.texts) + "\n\n")
        return ids

    def text(self, k: int) -> str:
        return self.tok.decode(self.base_ids(k + 8)[:k])


def fit(tok: Tok, base: list, make_messages, target: int, k0: int):
    """Choose a slice length k of `base` so that the prompt has `target` tokens."""
    k = max(1, min(k0, len(base)))
    best = None
    for _ in range(10):
        msgs = make_messages(tok.decode(base[:k]))
        n = tok.prompt_len(msgs)
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (msgs, n)
        if n == target:
            break
        k = max(1, min(len(base), k + (target - n)))
    return best


def lcp(a: list, b: list) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# ------------------------------------------------------------------ profiles
def gen_api(tok, src_h, C, n, rng):
    units = []
    for i in range(n):
        base = src_h.base_ids(C + 64)
        msgs, L = fit(tok, base, lambda t: [{"role": "system", "content": S.API_SYSTEM_PROMPT},
                                           {"role": "user", "content": t}], C, C - 60)
        units.append({"unit_id": f"api-{i}", "requests": [{"messages": msgs, "prompt_tokens": L}]})
    return units


def agentic_prefix(tok, tools, C, rng):
    target = round(S.AGENTIC_PREFIX_SHARE * C)
    header = ("You are an autonomous software agent. You can call the tools described below. "
              "When a tool is needed, answer with a JSON object {\"name\": ..., \"arguments\": ...}. "
              "Available tools:\n\n")
    text = header
    while len(tok.ids(text)) < target + 64:
        text += rng.choice(tools) + "\n\n"
    base = tok.ids(text)
    k = target
    for _ in range(10):
        sys_content = tok.decode(base[:k])
        n = len(tok.prompt_ids([{"role": "system", "content": sys_content}], gen=False))
        if n == target:
            break
        k += target - n
    return sys_content


def gen_agentic(tok, tools, tasks, src_g, C, n, rng, prefix):
    units = []
    for i in range(n):
        task = rng.choice(tasks)
        base = tok.ids(f"Task: {task}\n\nObservation from the previous tool call:\n") + src_g.base_ids(C)
        msgs, L = fit(tok, base, lambda t: [{"role": "system", "content": prefix},
                                           {"role": "user", "content": t}], C, C - round(S.AGENTIC_PREFIX_SHARE * C))
        units.append({"unit_id": f"agentic-{i}", "requests": [{"messages": msgs, "prompt_tokens": L}]})
    return units


def gen_chat(tok, src_h, src_g, C, n_sessions, rng):
    units = []
    for i in range(n_sessions):
        replies = [src_g.text(S.OUTPUT_TOKENS) for _ in range(S.CHAT_TURNS - 1)]
        followups = [src_h.text(S.CHAT_USER_TOKENS) for _ in range(S.CHAT_TURNS - 1)]

        def session(first_user: str):
            msgs = [{"role": "system", "content": S.API_SYSTEM_PROMPT}, {"role": "user", "content": first_user}]
            turns = [list(msgs)]
            for r, u in zip(replies, followups):
                msgs = msgs + [{"role": "assistant", "content": r}, {"role": "user", "content": u}]
                turns.append(list(msgs))
            return turns
        base = src_h.base_ids(C + 64)
        # D = mean(L_t) - L_1 does not depend on the first message; measure it once
        probe = session(tok.decode(base[:64]))
        lens = [tok.prompt_len(t) for t in probe]
        D = statistics.fmean(lens) - lens[0]
        target_first = round(C - D)
        msgs1, L1 = fit(tok, base, lambda t: session(t)[0], target_first, max(1, target_first - 60))
        first_user = msgs1[1]["content"]
        turns = session(first_user)
        units.append({"unit_id": f"chat-{i}",
                      "requests": [{"messages": t, "prompt_tokens": tok.prompt_len(t)} for t in turns]})
    return units


# ------------------------------------------------------------------ stats
def stats(tok, units, profile):
    lens, reuse, prev_first = [], [], None
    first_ids = None
    for u in units:
        prev = None
        for j, r in enumerate(u["requests"]):
            ids = tok.prompt_ids(r["messages"])
            lens.append(len(ids))
            if profile == "chat":
                ref = prev if prev is not None else prev_first
                reuse.append(lcp(ids, ref) if ref is not None else 0)
                if j == 0:
                    prev_first = ids
            else:
                reuse.append(lcp(ids, first_ids) if first_ids is not None else 0)
                if first_ids is None:
                    first_ids = ids
            prev = ids
    return {"n_units": len(units), "n_requests": len(lens), "prompt_tokens_mean": round(statistics.fmean(lens), 2),
            "prompt_tokens_min": min(lens), "prompt_tokens_max": max(lens),
            "reusable_prefix_share_pct": round(100.0 * sum(reuse) / sum(lens), 2)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(S.MODELS), required=True)
    ap.add_argument("--sharegpt", default=str(S.SOURCES_DIR / "sharegpt" / "ShareGPT_V3_unfiltered_cleaned_split.json"))
    ap.add_argument("--glaive", default=str(S.SOURCES_DIR / "glaive" / "glaive-function-calling-v2.json"))
    ap.add_argument("--profiles", default=",".join(S.PROFILES))
    ap.add_argument("--contexts", default=",".join(S.CONTEXTS))
    ap.add_argument("--pool", type=int, default=S.POOL_REQUESTS)
    ap.add_argument("--warmup-pool", type=int, default=S.WARMUP_POOL_REQUESTS)
    ap.add_argument("--seed", type=int, default=S.SEED)
    args = ap.parse_args(argv)

    tok = Tok(S.MODELS[args.model]["path"])
    human, gpt = load_sharegpt(Path(args.sharegpt))
    tools, tasks = load_glaive(Path(args.glaive))
    print(f"corpora: {len(human)} human, {len(gpt)} gpt texts, {len(tools)} tool schemas, {len(tasks)} tasks", flush=True)
    out_dir = S.WORKLOADS_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for context in args.contexts.split(","):
        C = S.CONTEXTS[context]
        for profile in args.profiles.split(","):
            prefix = None
            for warm in (False, True):
                rng = random.Random(f"{args.seed}-{args.model}-{profile}-{context}-{'warm' if warm else 'meas'}")
                src_h, src_g = Source(human, tok, rng), Source(gpt, tok, rng)
                n_req = args.warmup_pool if warm else args.pool
                if profile == "api":
                    units = gen_api(tok, src_h, C, n_req, rng)
                elif profile == "agentic":
                    if prefix is None:  # the same global prefix for measured and warm-up requests
                        prefix = agentic_prefix(tok, tools, C, random.Random(f"{args.seed}-{args.model}-prefix-{context}"))
                    units = gen_agentic(tok, tools, tasks, src_g, C, n_req, rng, prefix)
                else:
                    units = gen_chat(tok, src_h, src_g, C, n_req // S.CHAT_TURNS, rng)
                path = S.workload_file(args.model, profile, context, warmup=warm)
                with open(path, "w") as f:
                    for u in units:
                        f.write(json.dumps(u) + "\n")
                st = stats(tok, units, profile)
                st["target_mean_tokens"] = C
                manifest[path.name] = st
                print(f"{path.name}: {st}", flush=True)
                manifest_path.write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
