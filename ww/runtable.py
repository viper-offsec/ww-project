"""Run-table construction: full factorial on Llama, 8 corner cells on Qwen,
REPETITIONS randomized complete blocks (every block runs each cell once, in a
seeded random order). If the experiment is interrupted, every finished block is
a complete replicate of the design.
"""
import random

from ww import settings as S
from ww.runmetrics import DATA_COLUMNS


def make_block_shuffled(base_cls):
    class BlockShuffledRunTableModel(base_cls):
        def __init__(self, *args, seed: int = 0, **kwargs):
            kwargs["shuffle"] = False
            super().__init__(*args, **kwargs)
            self._ww_seed = seed
            self._ww_reps = kwargs.get("repetitions", 1)

        def generate_experiment_run_table(self):
            rows = super().generate_experiment_run_table()   # repetition-major order
            size = len(rows) // self._ww_reps
            rng = random.Random(self._ww_seed)
            out = []
            for j in range(self._ww_reps):
                block = rows[j * size:(j + 1) * size]
                rng.shuffle(block)
                out.extend(block)
            return out
    return BlockShuffledRunTableModel


def build(FactorModel, RunTableModel, repetitions: int = S.REPETITIONS, models=("llama", "qwen"),
          engines=None, profiles=None, contexts=None):
    """engines/profiles/contexts restrict the design (used only for short end-to-end tests)."""
    model = FactorModel("model", list(models))
    engine = FactorModel("engine", list(engines or S.ENGINES))
    profile = FactorModel("profile", list(profiles or S.PROFILES))
    context = FactorModel("context", list(contexts or S.CONTEXTS))
    exclude = []
    if "qwen" in models and not (engines or profiles or contexts):
        # Exclusions must be disjoint: experiment-runner deletes a row once per
        # matching rule, so a row matching two rules would remove a wrong row.
        drop_p = [p for p in S.PROFILES if p not in S.QWEN_CELLS["profiles"]]
        drop_c = [c for c in S.CONTEXTS if c not in S.QWEN_CELLS["contexts"]]
        if drop_p:
            exclude.append({model: ["qwen"], profile: drop_p})
        if drop_c:
            exclude.append({model: ["qwen"], profile: list(S.QWEN_CELLS["profiles"]), context: drop_c})
    cls = make_block_shuffled(RunTableModel)
    return cls(factors=[model, engine, profile, context], exclude_combinations=exclude,
               repetitions=repetitions, data_columns=list(DATA_COLUMNS), seed=S.RUN_ORDER_SEED)
