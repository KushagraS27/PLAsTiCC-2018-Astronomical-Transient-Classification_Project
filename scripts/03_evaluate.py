#!/usr/bin/env python3
"""03 - Run the full v2 evaluation suite and write the consolidated metrics.

Produces ``reports/metrics.json`` plus the candidate queues the dashboard serves.
Every headline number in the report comes from a locked split; the weight search
never sees one.

    python scripts/03_evaluate.py [--fast] [--skip-mismatched]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cne.config import load_config  # noqa: E402
from cne.logging import get_logger  # noqa: E402
from cne.pipeline import CNEPipeline  # noqa: E402

log = get_logger("evaluate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="200 bootstrap draws instead of 1000")
    parser.add_argument("--skip-mismatched", action="store_true")
    parser.add_argument("--skip-reweighted", action="store_true")
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs" / "default.yaml")
    pipeline = CNEPipeline(cfg, root=ROOT)
    metrics = pipeline.run_full_evaluation(
        do_mismatched=not args.skip_mismatched,
        do_reweighted=not args.skip_reweighted,
        bootstrap_draws=200 if args.fast else None,
    )
    for name, block in metrics["benchmarks"].items():
        m = block["metrics"]
        log.info("%-26s n=%-6d base=%.4f AUC=%.4f AP=%.4f P@50=%.3f lift@50=%.2f", name,
                 m["n_objects"], m["base_rate"], m["roc_auc"], m["average_precision"],
                 m["precision_at_k"].get("P@50", float("nan")), m["lift_at_k"].get("lift@50", float("nan")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
