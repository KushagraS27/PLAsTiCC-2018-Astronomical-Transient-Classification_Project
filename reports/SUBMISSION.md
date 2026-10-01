# SuperNova 2026 - Submission Package
## Problem Statement AI-05: AI-Based Discovery of Rare & Previously Unknown Astronomical Phenomena

**Project:** Cosmic Novelty Engine v2.0.0 (Aediles)
**Dataset:** PLAsTiCC (Zenodo 2539456)
**Seed:** 42 - Manifest: `cne-v2-splits` - Config: `configs/weights_v2.yaml`

---

## 1. Problem statement

Upcoming alert streams (LSST/Rubin) will deliver ~10 million transient alerts a
night. A small fraction are genuinely new kinds of astrophysical events. The task
is to rank alerts by how badly they are explained by *known* physics, so a human
astronomer spends their scarce review time on the most promising candidates.

The core difficulty is that "anomaly" has two confusable meanings:

1. **Astrophysically unusual** - the thing we want.
2. **Out of the reference domain** - a survey/photometry difference we must NOT
   mistake for #1.

Nearly every unsupervised anomaly detector conflates the two. This submission
makes that distinction the centre of the method, and proves it with measurement.

## 2. Innovation summary (what is new, and measured)

1. **Domain-matched known-physics prior.** Instead of generic anomaly detection,
   the engine first learns the 12 known populations (CV accuracy 0.826, macro AUC
   0.967) and ranks objects by how poorly any known class explains them. Novelty
   is defined *against* known physics, not against a density model.

2. **Multiplicative, suppress-only quality gating.** Bad data can only lower a
   score, never raise it. Verified by an artifact-injection suite
   (`scripts/05_artifact_safety.py`): five of six injected defect types push the
   object *down* the queue (mean rank gain -3.5 to -12.2).

3. **Adversarial domain validation** (`cne/adversarial.py`) - the diagnostic that
   won SETI Breakthrough Listen, adapted here. We train a classifier to separate
   reference from target on the same features the engine uses. **AUC 0.5438**: the
   populations are indistinguishable, so within the benchmark "novel" is *not* a
   proxy for "different domain". This is the failure mode that punished the
   runners-up in SETI, and we measure that we do not have it.

4. **Validation-only nested weight selection with a live leakage guard.** Weights
   are chosen on the validation split only; `LeakageGuard` locks the test split
   (locked_n=1,659, violations=0) and fails the run if any tuning stage reads it.
   Measured selection optimism is 0.048 ± 0.021, quantified not assumed.

5. **Base-rate honesty.** Every headline number carries dataset version, split
   manifest, seed, config id and base rate. We show that at deployment priors,
   precision is the meaningful metric and AUC is misleading
   (mismatched reference wins AUC 0.657 but loses P@10 0.30 vs 0.70).

## 3. Technical workflow

```
Alert stream -> Featuriser (401 features, 32 physical)
             -> Known-physics prior (12 classes, bootstrap ensemble)
             -> 9 measured evidence channels (4 active, 5 zero-weight audits)
             -> quality gate (multiplicative, suppress-only)
             -> uncertainty-aware abstention
             -> ranked queue + per-candidate explanation
```

Ranking (preserved from v1):
`novelty = sum_i w_i * evidence_i  x quality^0.5 x confidence^0.25 x (1+0.15*agreement)`
`confidence = (1 - uncertainty) * quality`

Reproducibility (offline, from the shipped bundle):

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                       # 233 passed
python scripts/03_evaluate.py                    # matched benchmarks from bundled train features
python scripts/06_figures.py                     # regenerate all 8 figures from metrics.json
python -m uvicorn api.main:app --port 8000       # live dashboard
```

## 4. Evaluation plan (how the numbers were produced)

| split | n | role |
|---|---|---|
| prior_fit | 4,621 | known-physics prior only |
| validation | 1,568 | weight selection only |
| test_matched | 1,659 | locked domain-matched benchmark |
| test_fullscale | 112,930 | locked cross-domain benchmark |

Headline (locked, base rate shown):

| benchmark | n | base rate | AUC | AP | P@50 | lift@50 |
|---|---|---|---|---|---|---|
| matched_baseline_v1 | 1,659 | 0.1073 | 0.6821 | 0.3546 | 0.68 | 6.34x |
| matched_nested_v2 | 1,659 | 0.1073 | 0.6883 | 0.3752 | 0.80 | 7.46x |
| fullscale_domain_matched | 112,930 | 0.0270 | 0.5525 | 0.0373 | 0.38 | 14.06x |

Bootstrap confidence intervals are included in `metrics.json`. Every figure in
`reports/figures/` is drawn from `metrics.json` or the frozen artefacts - none is
hand-typed.

## 5. Limitations (stated as measured, not tuned away)

1. **CaRT is not detectable with the available data.** CaRT is 68% of held-out
   objects (2,087 of 3,052) and scores enrichment 0.0 - because its median
   observation (4 detections, SNR 7.57, 3 bands) is statistically identical to a
   core-collapse supernova. This is a sampling limitation, verified, not a ranker
   bug. (See section 6 - the simulator shows the path forward.)
2. **Uncertainty is anti-correlated with novelty** (AUC 0.42). It feeds abstention
   only; it is never used as a novelty signal.
3. **The engine is domain-honest.** The full-scale stream returns `abstain`
   (mean PSI 0.47). We report that we are outside the validated domain rather than
   pretending otherwise.
4. **Featuriser reductions are float32 and not bit-reproducible**; benchmarks are
   stable to ~3 decimals, not exactly. Documented.
5. **Full-scale reproduction needs chunks 02/03** (~1.4 GB), not bundled for size.
   Matched-split results and all figures are fully reproducible offline.

## 6. Future work (the next win)

**Synthetic rare-transient simulator** (`cne/simulator.py`, 9 tests, offline). The
SETI-winning solution injected a simulated test-only signal into training. We built
the CNE analogue: a parameterised light-curve simulator for the withheld rare
classes (CaRT, KN), plus a known-like SNII control.

What it can and cannot honestly show:

- It **can** show the engine has the *capacity* to detect fast, rare transients
  when the data presents a distinguishing signal - the simulated CaRT/KN templates
  are separable from the SNII control by construction (fast rise/decay, red bands).
- It **cannot** show that real CaRT becomes detectable, because real CaRT's median
  observation (4 detections, SNR 7.57, 3 bands) is statistically identical to SNII.
  The simulator therefore frames CaRT correctly: a **measurement/coverage
  requirement**, not an architecture defect. No claim is made that it "fixes" CaRT.

The simulator's role is to make that distinction measurable and to give the next
team a tool to test detector capacity on separable rare classes without waiting
for new survey data.

---

**We never claim a discovery.** Every output is a *novelty-ranked candidate* that
is *poorly explained by current known populations* and *requires expert follow-up*.
