#!/usr/bin/env python3
"""02 - Freeze the protocol, fit the known-physics prior, select weights.

Order matters and is enforced here:

1. Freeze the split manifest (before anything is fitted).
2. Fit the known-physics prior on ``prior_fit`` only.
3. Fit the deliberately mismatched prior (for the degradation study).
4. Fit the density-ratio reweighted prior aimed at the full-scale stream.
5. Select channel weights on ``validation`` only, through a leakage guard.

Nothing in steps 2-5 may read a locked test object; step 5 asserts that.

    python scripts/02_train.py [--skip-mismatched] [--skip-reweighted]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cne.config import load_config  # noqa: E402
from cne.logging import get_logger  # noqa: E402
from cne.pipeline import CNEPipeline  # noqa: E402
from cne.version import stamp  # noqa: E402

log = get_logger("train")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-mismatched", action="store_true")
    parser.add_argument("--skip-reweighted", action="store_true")
    parser.add_argument("--force-manifest", action="store_true")
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs" / "default.yaml")
    pipeline = CNEPipeline(cfg, root=ROOT)
    pipeline.build_train_features()
    if (ROOT / cfg.data.processed_dir / "stream_features.parquet").exists():
        pipeline.build_stream_features()

    started = time.perf_counter()
    manifest = pipeline.make_manifest(force=args.force_manifest)
    log.info("manifest splits: %s", {k: len(v) for k, v in manifest.splits.items()})
    manifest.assert_disjoint("prior_fit", "validation", "test_matched")

    pipeline.fit_engine()
    log.info("known-physics prior: %s", pipeline.engine_.prior_.summary().as_dict())

    if not args.skip_mismatched:
        pipeline.fit_mismatched_engine()
    if not args.skip_reweighted and pipeline.stream_features_ is not None:
        pipeline.fit_reweighted_engine()

    weights = pipeline.select_weights()
    log.info("selected weights: %s", {k: v for k, v in weights.as_dict().items() if v > 0})
    log.info("leakage guard report: %s", pipeline.guard_.report())
    pipeline.guard_.assert_clean()

    artefacts = ROOT / cfg.data.artefacts_dir
    artefacts.mkdir(parents=True, exist_ok=True)
    summary = stamp(
        prior=pipeline.engine_.prior_.summary().as_dict(),
        splits={k: len(v) for k, v in manifest.splits.items()},
        selected_weights=weights.as_dict(),
        weight_search=getattr(pipeline, "weight_search_result_", None)
        and pipeline.weight_search_result_.as_dict(),
        leakage=pipeline.guard_.report(),
        elapsed_s=round(time.perf_counter() - started, 1),
    )
    (artefacts / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    log.info("training summary -> %s", artefacts / "training_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
