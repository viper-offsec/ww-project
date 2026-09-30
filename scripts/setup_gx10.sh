#!/usr/bin/env bash
# Environment setup used on the ASUS Ascent GX10 (NVIDIA GB10, Ubuntu 24.04, driver 580, CUDA 13.0).
# No sudo needed. Run from ~/greenlab_exp with this repository at ~/greenlab_exp/ww-experiment.
set -euo pipefail
cd "$HOME/greenlab_exp"
mkdir -p models workloads load_raw telemetry_raw analysis logs experiments

python3 -m venv .venv-tools  && .venv-tools/bin/pip install -U "huggingface_hub[cli]"
python3 -m venv .venv-vllm   && .venv-vllm/bin/pip install vllm==0.28.0 ninja
python3 -m venv .venv-sglang && .venv-sglang/bin/pip install sglang==0.5.19 ninja
python3 -m venv .venv-runner && .venv-runner/bin/pip install -r ww-experiment/requirements-runner.txt

# models (Llama is gated: run `.venv-tools/bin/hf auth login` first)
.venv-tools/bin/hf download meta-llama/Llama-3.1-8B-Instruct --local-dir models/Llama-3.1-8B-Instruct --exclude "original/*"
.venv-tools/bin/hf download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct

# text corpora for the workloads
.venv-tools/bin/hf download anon8231489123/ShareGPT_Vicuna_unfiltered ShareGPT_V3_unfiltered_cleaned_split.json \
    --repo-type dataset --local-dir load_raw/sharegpt
.venv-tools/bin/hf download glaiveai/glaive-function-calling-v2 glaive-function-calling-v2.json \
    --repo-type dataset --local-dir load_raw/glaive

# orchestration and measurement tools
git clone https://github.com/S2-group/experiment-runner.git
curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal && . "$HOME/.cargo/env"
git clone https://github.com/tdurieux/EnergiBridge.git
( cd EnergiBridge && patch -p1 < ../ww-experiment/patches/energibridge-arm-build.patch && cargo build -r )
