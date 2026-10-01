#!/usr/bin/env python3
"""01 - Featurise PLAsTiCC.

Two artefacts:

``data/processed/train_features.parquet``   the whole train split (7,848 objects,
    every label). Fits the known-physics prior and provides the domain-matched
    benchmark.
``data/processed/stream_features.parquet``  PLAsTiCC test chunk 01 harvested and
    featurised (32,926 objects). The full-scale, cross-domain benchmark.

    python scripts/01_featurise.py --train
    python scripts/01_featurise.py --stream
    python scripts/01_featurise.py --all --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cne.config import load_config  # noqa: E402
from cne.logging import get_logger  # noqa: E402
from cne.pipeline import CNEPipeline  # noqa: E402

log = get_logger("featurise")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-featurise even if the parquet exists")
    parser.add_argument("--max-stream", type=int, default=None,
                        help="cap the stream; scarce classes are always kept whole")
    args = parser.parse_args()

    do_train = args.train or args.all or not (args.train or args.stream)
    do_stream = args.stream or args.all
    cfg = load_config(ROOT / "configs" / "default.yaml")
    pipeline = CNEPipeline(cfg, root=ROOT)

    if do_train:
        started = time.perf_counter()
        features = pipeline.build_train_features(force=args.force)
        log.info("train features %s in %.1fs", features.shape, time.perf_counter() - started)

    if do_stream:
        started = time.perf_counter()
        features = pipeline.build_stream_features(max_objects=args.max_stream, force=args.force)
        log.info("stream features %s in %.1fs", features.shape, time.perf_counter() - started)
        log.info("stream class counts: %s", pipeline.stream_labels_.value_counts().sort_index().to_dict())

    log.info("feature count: %d (quality %d, astrophysical %d)",
             len(pipeline.featuriser_.feature_names()),
             len(pipeline.featuriser_.quality_columns),
             len(pipeline.featuriser_.astrophysical_columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
