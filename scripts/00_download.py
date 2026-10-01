#!/usr/bin/env python3
"""00 - Download the PLAsTiCC files this project needs.

Uses the ``?download=1`` URL form. The ``/files/{name}/content`` form returns
92-byte stubs, which is a trap that cost real time in CNE v1.

    python scripts/00_download.py            # train + test chunk 01
    python scripts/00_download.py --train-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cne.config import load_config  # noqa: E402
from cne.data import download_file, sha256_of_file  # noqa: E402
from cne.logging import get_logger  # noqa: E402

log = get_logger("download")

TRAIN_FILES = ("plasticc_train_metadata.csv.gz", "plasticc_train_lightcurves.csv.gz")
TEST_FILES = ("plasticc_test_metadata.csv.gz", "plasticc_test_lightcurves_01.csv.gz")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-only", action="store_true", help="skip the 300 MB of test data")
    parser.add_argument("--checksums", action="store_true", help="print sha256 of every file present")
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs" / "default.yaml")
    raw = ROOT / cfg.data.raw_dir
    files = list(TRAIN_FILES) + ([] if args.train_only else list(TEST_FILES))
    for name in files:
        path = download_file(name, raw)
        log.info("%s -> %s (%.1f MB)", name, path, path.stat().st_size / 1e6)
    if args.checksums:
        for name in files:
            path = raw / name
            if path.exists():
                log.info("sha256 %s %s", sha256_of_file(path)[:16], name)
    log.info("download complete: %d files in %s", len(files), raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
