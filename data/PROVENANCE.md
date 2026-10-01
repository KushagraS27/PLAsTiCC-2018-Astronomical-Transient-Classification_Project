# Data cache status — READ BEFORE TRUSTING THE NUMBERS

`reports/metrics.json` describes an evaluation over **112,930** full-scale objects.
The cached parquets in this directory were rebuilt after that run from
**PLAsTiCC test chunk 01 only** (chunks 02 and 03, ~1.4 GB, are not present in
this workspace).

| file | objects | status |
|---|---|---|
| `train_features.parquet` | 7,848 | complete — matches the run |
| `train_metadata.parquet` | 7,848 | complete |
| `stream_features.parquet` | 2,106 | **partial** — a subsample of chunk 01, used only to run the artifact-safety suite |
| `lightcurve_cache.parquet` | 2,274 | **partial** — covers 185 of the 400 dashboard candidates (46.2%) |

Consequences:

- The matched-split benchmark (1,659 objects) and everything derived from the
  train split are fully reproducible here.
- The **full-scale** benchmark (112,930 objects) is **not** reproducible without
  re-downloading chunks 02 and 03 and re-running `01_featurise.py --stream`.
- `stream_features.parquet` must not be presented as the full-scale stream. It is
  a 2,106-object working set.
- The dashboard's light-curve panel will show "no light curve" for the 215
  candidates whose photometry lived in chunks 02/03. Ranking, quality, tiers and
  explanations are unaffected — those come from the frozen artefacts in
  `data/artefacts/`, which are complete.

To restore full coverage:

```bash
python scripts/00_download.py          # re-fetches all chunks
python scripts/01_featurise.py --all   # rebuilds both feature sets
python scripts/03_evaluate.py          # regenerates metrics.json
```
