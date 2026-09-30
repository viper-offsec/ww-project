"""Experiment Runner configuration: vLLM vs SGLang energy/performance study.

Kept deliberately thin: experiment-runner checksums this file to allow resuming
an interrupted experiment, so all run logic lives in the ww/ package.

Run (from ~/greenlab_exp):
  .venv-runner/bin/python experiment-runner/experiment-runner/ ww-experiment/experiment/RunnerConfig.py
Environment:
  WW_EXPERIMENT_NAME  output folder name (default: ww_vllm_sglang_gx10)
  WW_REPETITIONS      repetitions per cell (default: settings.REPETITIONS)
  WW_MODELS           comma-separated subset of models (default: llama,qwen)
"""
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ConfigValidator.Config.Models.FactorModel import FactorModel  # noqa: E402
from ConfigValidator.Config.Models.OperationType import OperationType  # noqa: E402
from ConfigValidator.Config.Models.RunnerContext import RunnerContext  # noqa: E402
from ConfigValidator.Config.Models.RunTableModel import RunTableModel  # noqa: E402
from EventManager.EventSubscriptionController import EventSubscriptionController  # noqa: E402
from EventManager.Models.RunnerEvents import RunnerEvents  # noqa: E402

from ww import hooks, runtable  # noqa: E402
from ww import settings as S  # noqa: E402


class RunnerConfig:
    ROOT_DIR = REPO_ROOT

    name: str = os.environ.get("WW_EXPERIMENT_NAME", "ww_vllm_sglang_gx10")
    results_output_path: Path = S.RESULTS_DIR
    operation_type: OperationType = OperationType.AUTO
    time_between_runs_in_ms: int = 0          # cool-down is handled in stop_run (temperature-based)

    def __init__(self):
        EventSubscriptionController.subscribe_to_multiple_events([
            (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
            (RunnerEvents.BEFORE_RUN, self.before_run),
            (RunnerEvents.START_RUN, self.start_run),
            (RunnerEvents.START_MEASUREMENT, self.start_measurement),
            (RunnerEvents.INTERACT, self.interact),
            (RunnerEvents.STOP_MEASUREMENT, self.stop_measurement),
            (RunnerEvents.STOP_RUN, self.stop_run),
            (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
            (RunnerEvents.AFTER_EXPERIMENT, self.after_experiment),
        ])
        self.run_table_model = None

    def create_run_table_model(self) -> RunTableModel:
        def env_list(name):
            v = os.environ.get(name)
            return tuple(v.split(",")) if v else None
        reps = int(os.environ.get("WW_REPETITIONS", S.REPETITIONS))
        models = env_list("WW_MODELS") or ("llama", "qwen")
        self.run_table_model = runtable.build(FactorModel, RunTableModel, repetitions=reps, models=models,
                                              engines=env_list("WW_ENGINES"), profiles=env_list("WW_PROFILES"),
                                              contexts=env_list("WW_CONTEXTS"))
        return self.run_table_model

    def before_experiment(self) -> None:
        hooks.before_experiment(self.experiment_path)

    def before_run(self) -> None:
        pass

    def start_run(self, context: RunnerContext) -> None:
        hooks.start_run(context)

    def start_measurement(self, context: RunnerContext) -> None:
        hooks.start_measurement(context)

    def interact(self, context: RunnerContext) -> None:
        hooks.interact(context)

    def stop_measurement(self, context: RunnerContext) -> None:
        hooks.stop_measurement(context)

    def stop_run(self, context: RunnerContext) -> None:
        hooks.stop_run(context)

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, Any]]:
        return hooks.populate_run_data(context)

    def after_experiment(self) -> None:
        pass

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path: Path = None
