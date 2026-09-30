"""Central configuration of the WattWatchers vLLM-vs-SGLang energy experiment.

Every constant that shapes the experiment lives here, so the report, the run
table and the code cannot drift apart. Paths default to ~/greenlab_exp on the
testbed and can be overridden with the WW_EXP_ROOT environment variable.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = Path(os.environ.get("WW_EXP_ROOT", str(Path.home() / "greenlab_exp")))
MODELS_DIR = EXP_ROOT / "models"
WORKLOADS_DIR = EXP_ROOT / "workloads"
SOURCES_DIR = EXP_ROOT / "load_raw"          # raw text corpora used to build workloads
RESULTS_DIR = EXP_ROOT / "experiments"
LOGS_DIR = EXP_ROOT / "logs"

# ---------------------------------------------------------------- subjects
MODELS = {
    "llama": {"path": MODELS_DIR / "Llama-3.1-8B-Instruct", "served_name": "llama"},
    "qwen": {"path": MODELS_DIR / "Qwen2.5-7B-Instruct", "served_name": "qwen"},
}

# ---------------------------------------------------------------- factors
ENGINES = ["vllm", "sglang"]
PROFILES = ["api", "chat", "agentic"]
CONTEXTS = {"small": 512, "medium": 2048, "large": 8192}   # target mean input tokens / request
QWEN_CELLS = {"profiles": ["api", "agentic"], "contexts": ["small", "large"]}  # replication corners
REPETITIONS = 10
RUN_ORDER_SEED = 20260927

# ---------------------------------------------------------------- fixed parameters
OUTPUT_TOKENS = 128
CONCURRENCY = 8
MAX_RUNNING_SEQS = 16             # scheduler batch limit in both engines (>= CONCURRENCY)
KV_POOL_TOKENS = 81_920           # identical KV-cache capacity in both engines
VLLM_BLOCK_SIZE = 16
MAX_MODEL_LEN = 16_384
SEED = 0

# workload shape
API_SYSTEM_PROMPT = ("You are a helpful and concise assistant. Answer the user's request "
                     "accurately and completely, using plain language.")
CHAT_TURNS = 4
CHAT_USER_TOKENS = 64             # every follow-up user message
AGENTIC_PREFIX_SHARE = 0.75       # share of the prompt taken by the global tool prefix
POOL_REQUESTS = 480               # requests generated per cell (measured phase draws from these)
WARMUP_POOL_REQUESTS = 160

# ---------------------------------------------------------------- run protocol
WARMUP_SECONDS = 60
MIN_WINDOW_SECONDS = 180
PRE_ROLL_S = 3
POST_ROLL_S = 3
COOLDOWN_MIN_S = 60
COOLDOWN_MAX_S = 300
COOLDOWN_TEMP_DELTA_C = 3.0
SAMPLE_INTERVAL_S = 0.1           # NVML energy/power sampler (10 Hz)
METRICS_SCRAPE_INTERVAL_S = 0.5
ENGINE_START_TIMEOUT_S = 1200
REQUEST_TIMEOUT_S = 1800
GUARD_CPU_PCT = 30.0              # a foreign process above this CPU share contaminates a run
GUARD_MAX_WAIT_S = 600

# ---------------------------------------------------------------- environments
VENVS = {
    "vllm": EXP_ROOT / ".venv-vllm",
    "sglang": EXP_ROOT / ".venv-sglang",
    "runner": EXP_ROOT / ".venv-runner",
}
PORTS = {"vllm": 8000, "sglang": 30000}
ENERGIBRIDGE_BIN = Path(os.environ.get(
    "WW_ENERGIBRIDGE", str(EXP_ROOT / "EnergiBridge" / "target" / "release" / "energibridge")))


def runner_python() -> str:
    p = VENVS["runner"] / "bin" / "python"
    return str(p) if p.exists() else "python3"


def cell_name(profile: str, context: str) -> str:
    return f"{profile}_{context}"


def workload_file(model: str, profile: str, context: str, warmup: bool = False) -> Path:
    suffix = "_warmup" if warmup else ""
    return WORKLOADS_DIR / model / f"{cell_name(profile, context)}{suffix}.jsonl"


N_PER_CELL_FILE = WORKLOADS_DIR / "n_per_cell.json"   # written by scripts/pilot.py
