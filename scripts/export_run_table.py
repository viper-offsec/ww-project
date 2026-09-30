"""Export the planned run table (same generator experiment-runner uses) to CSV.

  python scripts/export_run_table.py --er-dir <experiment-runner/experiment-runner> [--out data/Run_Table_planned.csv]
"""
import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--er-dir", required=True, help="path to experiment-runner/experiment-runner")
    ap.add_argument("--out", default=str(REPO / "data" / "Run_Table_planned.csv"))
    args = ap.parse_args(argv)
    sys.path.insert(0, args.er_dir)
    from ConfigValidator.Config.Models.FactorModel import FactorModel
    from ConfigValidator.Config.Models.RunTableModel import RunTableModel
    from ww import runtable
    rows = runtable.build(FactorModel, RunTableModel).generate_experiment_run_table()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["__done"] = r["__done"].name
            w.writerow({k: ("" if v == " " else v) for k, v in r.items()})
    n_cells = len({(r["model"], r["engine"], r["profile"], r["context"]) for r in rows})
    print(f"{len(rows)} runs ({n_cells} cells) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
