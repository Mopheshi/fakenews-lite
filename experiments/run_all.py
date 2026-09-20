"""Run every experiment in dependency order.

``run_main.py`` must go first: it writes the fitted models that the efficiency
benchmark loads and the test-set prediction arrays that the ablation and the
significance tests consume. The rest are independent of each other.

Expect several hours on a CPU-only machine. Each step is skipped if its result
file already exists, so the sequence can be interrupted and resumed.

    python experiments/run_all.py
    python experiments/run_all.py --only main pos_ablation
    python experiments/run_all.py --force
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

from _report import configure_logging
from fakenews.config import RESULTS_DIR

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent

#: name -> (script, result file that marks it complete)
STEPS: dict[str, tuple[str, str]] = {
    "main": ("run_main.py", "main_results.json"),
    "pos_ablation": ("run_pos_ablation.py", "pos_ablation.json"),
    "efficiency": ("run_efficiency.py", "efficiency.json"),
    "svd_sweep": ("run_svd_sweep.py", "svd_sweep.json"),
    "leakage": ("run_leakage.py", "source_leakage.json"),
    "seeds": ("run_seeds.py", "seed_stability.json"),
    "xgb_tuning": ("run_xgb_tuning.py", "xgboost_tuning.json"),
    "source_holdout": ("run_source_holdout.py", "source_holdout.json"),
    "error_analysis": ("run_error_analysis.py", "error_analysis.json"),
    "cross_dataset": ("run_cross_dataset.py", "cross_dataset.json"),
}

#: Tuning is slow, and the cross-dataset probe needs a corpus that is not part
#: of the standard setup, so both are opt-in.
DEFAULT_STEPS = (
    "main",
    "pos_ablation",
    "efficiency",
    "svd_sweep",
    "leakage",
    "seeds",
    "source_holdout",
    "error_analysis",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=list(STEPS), default=list(DEFAULT_STEPS))
    parser.add_argument("--force", action="store_true", help="Re-run steps whose results exist.")
    args = parser.parse_args()

    configure_logging()

    failures = []
    for name in args.only:
        script, marker = STEPS[name]

        if (RESULTS_DIR / marker).exists() and not args.force:
            logger.info("[%s] already complete (%s); skipping", name, marker)
            continue

        logger.info("[%s] starting %s", name, script)
        started = time.perf_counter()
        completed = subprocess.run([sys.executable, str(HERE / script)], check=False)
        elapsed = time.perf_counter() - started

        if completed.returncode == 0:
            logger.info("[%s] finished in %.1f min", name, elapsed / 60)
        else:
            logger.error("[%s] FAILED after %.1f min (exit %d)", name, elapsed / 60, completed.returncode)
            failures.append(name)

    if failures:
        logger.error("Failed steps: %s", ", ".join(failures))
        return 1

    logger.info("All requested experiments complete. Results in %s", RESULTS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
