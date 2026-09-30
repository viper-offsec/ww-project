"""Mark invalid runs (valid != 1) as TODO again so experiment-runner re-executes
them on the next start. The original run table is backed up first.

  python scripts/requeue_invalid.py ~/greenlab_exp/experiments/ww_vllm_sglang_gx10 [--dry-run]
"""
import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ww.runmetrics import DATA_COLUMNS  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment_dir")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    table = Path(args.experiment_dir) / "run_table.csv"
    with open(table) as f:
        rd = csv.DictReader(f)
        fields, rows = rd.fieldnames, list(rd)
    requeue = [r for r in rows if r["__done"] == "DONE" and str(r.get("valid", "")).strip() not in ("1", "1.0")]
    for r in requeue:
        print(f"requeue {r['__run_id']}: {r.get('invalid_reason', '')}")
        if not args.dry_run:
            r["__done"] = "TODO"
            for k in DATA_COLUMNS:
                if k in r:
                    r[k] = " "
    if requeue and not args.dry_run:
        shutil.copy(table, table.with_suffix(f".backup_{time.strftime('%Y%m%d_%H%M%S')}.csv"))
        with open(table, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    done = sum(r["__done"] == "DONE" for r in rows)
    print(f"{len(requeue)} runs requeued; {done}/{len(rows)} DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
