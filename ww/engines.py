"""Start, health-check and stop the two inference engines with matched settings.

Both engines get the same model, dtype, context length, KV-cache capacity (in
tokens), seed and prefix caching; everything else stays at the engine defaults.
"""
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from ww import settings as S


def base_url(engine: str) -> str:
    return f"http://127.0.0.1:{S.PORTS[engine]}"


def engine_command(engine: str, model: str) -> list:
    m = S.MODELS[model]
    port = str(S.PORTS[engine])
    if engine == "vllm":
        return [
            str(S.VENVS["vllm"] / "bin" / "vllm"), "serve", str(m["path"]),
            "--served-model-name", m["served_name"],
            "--host", "127.0.0.1", "--port", port,
            "--dtype", "bfloat16",
            "--max-model-len", str(S.MAX_MODEL_LEN),
            "--block-size", str(S.VLLM_BLOCK_SIZE),
            "--num-gpu-blocks-override", str(S.KV_POOL_TOKENS // S.VLLM_BLOCK_SIZE),
            "--enable-prefix-caching",
            "--max-num-seqs", str(S.MAX_RUNNING_SEQS),
            "--enable-prompt-tokens-details",      # report cached prompt tokens in usage
            "--generation-config", "vllm",         # ignore model-specific sampling defaults
            "--seed", str(S.SEED),
        ]
    if engine == "sglang":
        return [
            str(S.VENVS["sglang"] / "bin" / "python"), "-m", "sglang.launch_server",
            "--model-path", str(m["path"]),
            "--served-model-name", m["served_name"],
            "--host", "127.0.0.1", "--port", port,
            "--dtype", "bfloat16",
            "--context-length", str(S.MAX_MODEL_LEN),
            "--max-total-tokens", str(S.KV_POOL_TOKENS),
            "--max-running-requests", str(S.MAX_RUNNING_SEQS),
            "--cuda-graph-max-bs", str(S.MAX_RUNNING_SEQS),
            "--random-seed", str(S.SEED),
            "--enable-metrics",
            "--enable-cache-report",              # report cached prompt tokens in usage
        ]
    raise ValueError(engine)


def start_engine(engine: str, model: str, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = engine_command(engine, model)
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    # FlashInfer JIT-compiles kernels on first use: it needs ninja (venv) and nvcc (CUDA toolkit)
    path = os.pathsep.join([str(S.VENVS[engine] / "bin"), f"{cuda_home}/bin", os.environ.get("PATH", "")])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", PYTHONUNBUFFERED="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUDA_HOME=cuda_home, PATH=path)
    logf = open(log_path, "w")
    logf.write("# " + " ".join(cmd) + "\n")
    logf.flush()
    # own session -> the whole engine process tree can be signalled as one group
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)


def _get(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read()


def is_ready(engine: str) -> bool:
    try:
        status, body = _get(base_url(engine) + "/v1/models")
        return status == 200 and b"data" in body
    except Exception:
        return False


def wait_ready(engine: str, proc: subprocess.Popen, timeout: float = S.ENGINE_START_TIMEOUT_S) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"{engine} exited during start-up with code {proc.returncode}")
        if is_ready(engine):
            return time.time() - t0
        time.sleep(1.0)
    raise TimeoutError(f"{engine} not ready after {timeout}s")


def port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def stop_engine(proc: subprocess.Popen, engine: str, timeout: float = 60.0) -> None:
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=30)
    if proc is not None:
        # stragglers that left the group (rare) are killed via the process group id
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    t0 = time.time()
    while port_in_use(S.PORTS[engine]) and time.time() - t0 < 60:
        time.sleep(1)


def kill_leftovers() -> None:
    """Kill engine servers left over from an interrupted run (same user only)."""
    import psutil
    me = os.getuid()
    for p in psutil.process_iter(["pid", "uids", "cmdline"]):
        try:
            if p.info["uids"].real != me:
                continue
            cmd = " ".join(p.info["cmdline"] or [])
            if ("vllm serve" in cmd or "sglang.launch_server" in cmd) and str(S.MODELS_DIR) in cmd:
                p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
            continue
    for engine in S.ENGINES:
        t0 = time.time()
        while port_in_use(S.PORTS[engine]) and time.time() - t0 < 60:
            time.sleep(1)


def engine_versions() -> dict:
    out = {}
    for engine, pkg in (("vllm", "vllm"), ("sglang", "sglang")):
        py = S.VENVS[engine] / "bin" / "python"
        try:
            v = subprocess.run([str(py), "-c",
                                f"import importlib.metadata as m, torch; print(m.version('{pkg}'), torch.__version__, torch.version.cuda)"],
                               capture_output=True, text=True, timeout=120).stdout.strip()
        except Exception as e:  # noqa: BLE001
            v = f"unavailable ({e})"
        out[engine] = v
    return out


if __name__ == "__main__":
    print(json.dumps(engine_versions(), indent=2))
