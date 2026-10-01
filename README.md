# Cosmic Novelty Engine v2.0.0 — *Aediles*

Candidate-anomaly triage for transient surveys, built on the real PLAsTiCC
dataset (Zenodo 2539456).

**This system never claims a discovery.** Every output is a *candidate anomaly*
that is *poorly explained by current known populations* and *requires expert
follow-up*. The string `NEW DISCOVERY CONFIRMED` is forbidden in this codebase
and is enforced by a test.

---

## Run it

```bash
pip install -r requirements.txt

cd web && npm install && npm run build && cd ..

python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
# dashboard:  http://localhost:8000
# API docs:   http://localhost:8000/docs
```

> **Open `http://localhost:8000` in your browser - NOT `http://0.0.0.0:8000`.**
> `0.0.0.0` only tells the server which interfaces to listen on; browsers cannot
> navigate to it (you will get `ERR_ADDRESS_INVALID`). Use `localhost` or
> `127.0.0.1` instead.


The API serves the frozen artefacts in `data/artefacts/` and `reports/`. It never
retrains on request — a service whose ranking changes between page loads is not
auditable.

Dev mode with hot reload (proxies `/api` to :8000):

```bash
cd web && npm run dev
```

## Reproduce the numbers

```bash
python scripts/00_download.py            # PLAsTiCC from Zenodo (~1.7 GB)
python scripts/01_featurise.py --all     # train + stream features, LC cache
python scripts/02_train.py               # known-physics prior
python scripts/03_evaluate.py            # -> reports/metrics.json
python scripts/04_early_detection.py     # truncated-lightcurve lower bound
python scripts/05_artifact_safety.py     # injected-corruption gate test
python -m pytest tests/ -q               # 228 tests
```

Full evaluation takes ~22 min on 2 cores / 2 GB RAM.

## Results (measured, `reports/metrics.json`)

Domain-matched locked test split, n = 1,659, base rate 0.1073:

| ranking | AUC [95% CI] | AP | P@10 | P@50 | lift@50 |
|---|---|---|---|---|---|
| `baseline_v1` | 0.6821 | 0.3546 | 1.00 | 0.68 | 6.34× |
| `nested_v2` | **0.6883** | **0.3752** | 1.00 | **0.80** | **7.46×** |

Full-scale cross-domain stream, n = 112,930, base rate 0.0270:

| reference | AUC | AP | P@10 | lift@10 |
|---|---|---|---|---|
| unweighted train | 0.5447 | 0.0350 | 0.60 | 22.2× |
| density-ratio reweighted | 0.5525 | 0.0373 | **0.70** | **25.9×** |
| deliberately mismatched | 0.6573 | 0.0604 | 0.30 | 11.1× |

Known-physics prior: CV accuracy 0.8258, macro ROC-AUC 0.9668, 12 classes.

### The honest negative result

**CaRT (class 993) is not detected.** It is 2,087 of the 3,052 held-out objects
(68%) at full scale and scores enrichment 0.0 — recall@200 of 0.000. This is not a
bug in the ranker: CaRT's median observation is **4 detections at SNR 7.57 across
3 bands, statistically identical to SNII (4 / 7.57 / 3)**. The abundant classes
are abstained at nearly the same rate (SNIa 0.679, SNII 0.687, CaRT 0.752), so the
quality gate is not the cause. With four detections, a calcium-rich fast transient
is not separable from a poorly sampled core-collapse supernova in this feature
space. Reported as measured, not tuned away.

Where the data *does* support separation, the engine performs well:
**muLens-Single 71.4× enrichment** (31 of 35 objects reach the top 200) and
**muLens-Binary 58.6×**.

### Why the mismatched reference has higher AUC

The 2-class (SNIa + AGN) prior flags far more objects as unusual, broadening
recall and raising whole-queue AUC. But its top-of-queue precision is much worse
(P@10 0.30 vs 0.70). **AUC is the misleading metric here**; precision at the
review budget is the operationally meaningful one. Never quote AUC without the
base rate.

## Important caveat on base rate

The 2.70% full-scale base rate is a **sampling artifact**. Chunks 02/03 were
thinned by `rare_preserving_subsample` (40,000 of 345,997 objects each), which
keeps every rare-class object by design. Chunk 01, taken whole, has a **0.60%**
base rate. Use the base-rate stress table (`stress_fullscale` in
`reports/metrics.json`), not the raw aggregate, for deployment expectations.

## Architecture

```
cne/            23 modules: physics, features, classifier, novelty, ranking,
                quality, uncertainty, domain, stress, artifacts, adapters,
                manifests, taxonomy, explain, evaluation, experiments
api/main.py     FastAPI service (10 endpoints, v1 surface preserved)
web/            React + Vite dashboard (built to web/dist)
scripts/        00 download → 05 artifact safety
tests/          228 passed (data/processed/train_features.parquet is bundled, so nothing skips)
reports/        RESEARCH_REPORT.md, metrics.json, manifests/, weight_selection.json
data/artefacts/ candidate queue + explanations
```

Ranking formula (preserved from v1):

```
novelty    = Σ wᵢ·evidenceᵢ × quality^0.5 × confidence^0.25 × (1 + 0.15·agreement)
confidence = (1 − uncertainty) · quality
```

Quality gating is **multiplicative and suppress-only** — bad data can never raise
a novelty score.

## Design invariants

- Novelty is never claimed from anomaly detection, autoencoders, similarity or
  clustering alone. Those remain **zero-weight audit channels**, measured and
  reported, never deleted.
- Held-out novel class labels never reach fitting, weight selection, threshold
  tuning or feature selection. Enforced by `LeakageGuard`, which fails the run if
  a locked-test object is read during tuning.
- Every benchmark carries dataset version, split manifest, seed, config id and
  base rate.
- Negative results are reported as measured.

See `CHANGELOG.md` for the 20 genuine bugs found and fixed during this rebuild.
