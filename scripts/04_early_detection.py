#!/usr/bin/env python3
"""04 - Early-detection lower bound.

Measures how much of the ranking's power survives when each object is scored
from only its first N days of photometry.

The models are deliberately NOT refit per window. Refitting on truncated light
curves would produce better numbers, but it would also mean training on a
feature distribution the deployed model never sees, and it would make the
comparison against the full-length result a comparison between two different
models. Scoring a fixed model on truncated views is a legitimate lower bound and
is leakage-free by construction: no future observation can enter training.

    python scripts/04_early_detection.py
    python scripts/04_early_detection.py --windows 15 30 60
    python scripts/04_early_detection.py --max-objects 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cne.experiments import EARLY_WINDOWS, early_detection_study  # noqa: E402
from cne.logging import get_logger  # noqa: E402
from cne.pipeline import CNEPipeline  # noqa: E402
from cne.ranking import RankingWeights  # noqa: E402
from cne.version import stamp  # noqa: E402

log = get_logger("early")



def load_weights(path):
    """Read the frozen v2 weight config and return a RankingWeights.

    ``RankingWeights`` has no ``load`` classmethod - the file is a plain YAML
    mapping under a ``weights:`` key (see ``NestedWeightSelector.write_config``).
    Both 04 and 05 called a nonexistent ``RankingWeights.load`` and crashed.
    """
    import yaml

    payload = yaml.safe_load(Path(path).read_text())
    return RankingWeights({k: float(v) for k, v in payload["weights"].items()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=int, nargs="*", default=list(EARLY_WINDOWS),
                        help="truncation windows in days (default: %(default)s)")
    parser.add_argument("--max-objects", type=int, default=None,
                        help="cap the scored stream (each window re-featurises everything)")
    args = parser.parse_args()

    pipeline = CNEPipeline()
    features = pipeline.build_stream_features(max_objects=args.max_objects)
    # fit_engine() reads self.manifest_.split("prior_fit") and indexes
    # self.train_features_, neither of which exists yet at this point. Without
    # these two calls the script dies with AttributeError on the first line that
    # touches them - the same defect fixed in 05_artifact_safety.py.
    pipeline.make_manifest()
    pipeline.build_train_features()
    engine = pipeline.fit_engine()
    weights = load_weights(pipeline.root / "configs" / "weights_v2.yaml")
    # Each window re-featurises the whole stream, so the photometry has to be
    # resident. Load only the objects that were actually scored.
    lightcurves = pipeline.lc_cache_.frame(object_ids=features["object_id"].tolist())
    if lightcurves is None or not len(lightcurves):
        log.error("no light-curve cache; run scripts/01_featurise.py first")
        return 1
    log.info("loaded %d rows for %d objects", len(lightcurves), lightcurves["object_id"].nunique())

    started = time.perf_counter()
    results = early_detection_study(pipeline, features, lightcurves, engine, weights, args.windows)
    results["provenance"] = stamp(
        config_id=pipeline.cfg.config_id,
        config_fingerprint=pipeline.cfg.fingerprint(),
        manifest=pipeline.manifest_.manifest_id,
        dataset="PLAsTiCC",
        windows=args.windows,
    )
    results["elapsed_s"] = round(time.perf_counter() - started, 1)

    out = pipeline.reports / "early_detection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info("wrote %s", out)

    def _fmt(value, width=7):
        return f"{value:>{width}.3f}" if isinstance(value, (int, float)) else f"{'n/a':>{width}}"

    header = f"{'window':>8} {'n':>7} {'novel':>6} {'AUC':>7} {'AP':>7} {'P@50':>7} {'R@200':>7}"
    print(header)
    print("-" * len(header))
    for row in results["windows"]:
        print(f"{row['window']:>8} {row['n_objects']:>7} {row['n_novel']:>6} "
              f"{_fmt(row['roc_auc'])} {_fmt(row['average_precision'])} "
              f"{_fmt(row['P@50'])} {_fmt(row['R@200'])}")
    print("\nLower bound: models are not refit per window. A refit model would score higher;")
    print("these numbers are what the deployed full-length model achieves on partial data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
