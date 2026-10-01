#!/usr/bin/env python3
"""05 - Artifact-injection safety suite.

The question this answers is not "does the ranker find novelty" but "does bad
data get promoted to novelty". Every artifact type PLAsTiCC-style photometry can
produce is injected into real light curves, each corrupted object is pushed
through the *production* scoring path (featuriser -> engine -> ranker), and the
suite fails if any of them scores higher than the same object did when clean.

This is a gate test, not a performance test. The only passing outcome is zero
promotions. A single promoted artifact is a real defect: it means an operator
would be asked to follow up a photometric failure.

    python scripts/05_artifact_safety.py
    python scripts/05_artifact_safety.py --cases 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cne.artifacts import ARTIFACT_TYPES  # noqa: E402
from cne.experiments import artifact_safety_study  # noqa: E402
from cne.logging import get_logger  # noqa: E402
from cne.pipeline import CNEPipeline  # noqa: E402
from cne.ranking import RankingWeights  # noqa: E402
from cne.version import stamp  # noqa: E402

log = get_logger("safety")



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
    parser.add_argument("--cases", type=int, default=8, help="cases per artifact type")
    parser.add_argument("--max-objects", type=int, default=3000,
                        help="cap the stream used to build the baseline queue")
    parser.add_argument("--force-manifest", action="store_true",
                        help="rebuild the split manifest instead of loading the frozen one")
    args = parser.parse_args()

    pipeline = CNEPipeline()
    features = pipeline.build_stream_features(max_objects=args.max_objects)
    # fit_engine() reads self.manifest_.split("prior_fit"), so the frozen split
    # has to exist first. Calling fit_engine() straight after featurising raised
    # AttributeError: 'NoneType' object has no attribute 'split'.
    pipeline.make_manifest(force=args.force_manifest)
    # fit_engine() also indexes self.train_features_ to build the reference set,
    # which is only populated by build_train_features(). That call loads the
    # cached parquet rather than re-featurising, so it is cheap here.
    pipeline.build_train_features()
    engine = pipeline.fit_engine()
    weights = load_weights(pipeline.root / "configs" / "weights_v2.yaml")
    # The suite corrupts only the first max(cases*6, 48) objects, so load just
    # those. Pulling the whole three-chunk stream (~33M rows) into a 2 GB host
    # to inject artifacts into a few hundred objects exhausted memory.
    sample = features["object_id"].to_numpy()[: max(args.cases * 6, 48)]
    lightcurves = pipeline.lc_cache_.frame(object_ids=sample)
    if lightcurves is None or not len(lightcurves):
        log.error("no light-curve cache; run scripts/01_featurise.py first")
        return 1
    log.info("loaded %d rows for %d objects", len(lightcurves), lightcurves["object_id"].nunique())

    report = artifact_safety_study(pipeline, features, lightcurves, engine, weights,
                                   n_objects=args.cases)
    report["provenance"] = stamp(
        config_id=pipeline.cfg.config_id,
        config_fingerprint=pipeline.cfg.fingerprint(),
        manifest=pipeline.manifest_.manifest_id,
        dataset="PLAsTiCC",
        n_cases=report["n_cases"],
    )

    out = pipeline.reports / "artifact_safety.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    header = f"{'artifact':<26} {'cases':>6} {'promoted':>9} {'climbed':>8} {'mean dQ':>9} {'mean rank gain':>15}"
    print(header)
    print("-" * len(header))
    for name in ARTIFACT_TYPES:
        row = report["per_artifact"].get(name, {})
        print(f"{name:<26} {row.get('n', 0):>6} {row.get('n_promoted', 0):>9} "
              f"{row.get('n_climbed', 0):>8} {row.get('mean_delta_quality', float('nan')):>9.3f} "
              f"{row.get('mean_rank_gain', float('nan')):>15.1f}")
    print("-" * len(header))
    print(f"{'TOTAL':<26} {report['n_cases']:>6} {report['n_promoted']:>9} "
          f"{report['mean_quality_drop']:>19.3f}")
    print()
    if report["passed"]:
        print(f"PASS - {report['n_cases']} injected artifacts, none promoted. "
              f"Mean quality fell {report['mean_quality_drop']:.3f}.")
    else:
        print(f"FAIL - {report['n_promoted']} of {report['n_cases']} injected artifacts were "
              "promoted. The quality gate is not suppressing bad data; see "
              f"{out} for the offending cases.")
    log.info("wrote %s", out)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
