#!/usr/bin/env bash
# Run the main experiment until every run in the run table is valid.
# experiment-runner resumes from the first unfinished run; invalid runs are
# re-queued (scripts/requeue_invalid.py) and executed again, at most 4 rounds.
#   tmux new-window -t ww -n main "bash ~/greenlab_exp/ww-experiment/scripts/run_main.sh"
set -uo pipefail
cd "$HOME/greenlab_exp"
NAME="${WW_EXPERIMENT_NAME:-ww_vllm_sglang_gx10}"
EXP="experiments/$NAME"
LOG="logs/main_$NAME.log"
for round in 1 2 3 4; do
  echo "=== round $round $(date '+%F %T')" | tee -a "$LOG"
  .venv-runner/bin/python experiment-runner/experiment-runner/ ww-experiment/experiment/RunnerConfig.py 2>&1 | tee -a "$LOG"
  n=$(.venv-runner/bin/python ww-experiment/scripts/requeue_invalid.py "$EXP" 2>&1 | tee -a "$LOG" | tail -1 | awk '{print $1}')
  echo "round $round: requeued ${n:-?} invalid runs" | tee -a "$LOG"
  [ "${n:-1}" = "0" ] && break
done
echo "MAIN_EXPERIMENT_FINISHED $(date '+%F %T')" | tee -a "$LOG"
