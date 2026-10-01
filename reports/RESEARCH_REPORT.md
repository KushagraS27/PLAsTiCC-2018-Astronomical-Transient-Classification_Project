# Cosmic Novelty Engine v2.0.0 — Research Report

**Codename:** Aediles · **Dataset:** PLAsTiCC (Zenodo 2539456) · **Seed:** 42
**Manifest:** `cne-v2-splits` · **Config:** `configs/default.yaml` + `configs/weights_v2.yaml`
**Metrics:** `reports/metrics.json` (runtime 1,329 s, 2 cores / 2 GB)

Every number below carries its dataset version, split manifest, seed, config id
and base rate. Negative results are reported as measured and were not tuned away.

---

## 1. Headline

The engine ranks candidate anomalies well when the reference population matches
the survey domain, and degrades badly when it does not. On the domain-matched
locked test split it reaches **AUC 0.6883** and **P@50 0.80**; on the full-scale
cross-domain stream it falls to **AUC 0.5525**.

Crucially, **the system found one held-out population reliably and completely
missed another**, and the reason is a property of the data, not of the model
(§5).

No output of this system is a confirmed discovery. Every candidate is *poorly
explained by current known populations* and *requires expert follow-up*.

## 2. Splits

Frozen before anything was fitted or tuned, at seed 42:

| split | n | role |
|---|---|---|
| `prior_fit` | 4,621 | known-physics prior only |
| `validation` | 1,568 | **the only objects weight selection may read** |
| `test_matched` | 1,659 | locked, domain-matched benchmark |
| `test_fullscale` | 112,930 | locked, cross-domain benchmark |

12 populations form the prior. Withheld entirely: KN (64), CaRT (993), ILOT
(992), PISN (994), muLens-Binary (991), and muLens-Single (6, only 151 training
examples — disclosed wherever it appears).

`LeakageGuard` locks `test_matched` (1,659 objects) and fails the run if any
tuning stage reads one.

### Disclosure: the persisted guard record is from a pre-fix run

`reports/metrics.json` records `leakage_guard.locked_split = "validation"` and
`locked_n = 0`. That is a **stale, vacuous record**. At the time this bundle was
written, `CNEPipeline.make_manifest()` returned early on the cached-manifest
path without rebuilding the guard, so the guard kept its manifest-less
`__init__` default, locked the *wrong* split, and held zero locked ids. With
`locked_n = 0`, `assert_clean()` could never fire. The run "passed" while
verifying nothing.

The defect is fixed at `cne/pipeline.py:288` and now covered by four tests.
Verified live against the frozen manifest:

```
BEFORE make_manifest -> locked_split='validation', locked_n=0
AFTER  make_manifest -> locked_split='test_matched', locked_n=1659
assert_clean() raised: AssertionError - locked 'test_matched' objects were
                       read during tuning: {'weight_search': [...]}
```

**Did leakage actually occur? No — but that is established independently of the
guard, not by it.** `weight_search` recorded exactly **1,568** ids seen, which
equals `len(validation)` exactly, and `validation ∩ test_matched = 0` in the
frozen manifest. So the selection demonstrably read only validation objects.

The honest framing: the *outcome* is clean and the *evidence* is now sound, but
the control that was supposed to supply that evidence was inert for this run.
`metrics.json` has not been regenerated, so its `leakage_guard` block should be
read as a record of the defect rather than as a clean result. Re-running
`scripts/03_evaluate.py` will write an armed record.

## 3. Known-physics prior

CV accuracy **0.8258**, macro ROC-AUC **0.9668**, 12 classes, 4,621 objects,
4-fold. This is the supervised ceiling the novelty ranking is built on top of.

## 4. Benchmarks

### 4.1 Domain-matched (locked, n = 1,659, base rate 0.1073)

| ranking | AUC | AP | P@10 | P@20 | P@50 | lift@50 |
|---|---|---|---|---|---|---|
| `baseline_v1` | 0.6821 | 0.3546 | 1.00 | 0.95 | 0.68 | 6.34× |
| `nested_v2` | **0.6883** | **0.3752** | 1.00 | 0.95 | **0.80** | **7.46×** |

Nested selection beats the frozen v1 weighting on every headline metric. The
improvement is modest and honest: +0.006 AUC, +0.021 AP, +0.12 P@50.

### 4.2 Full-scale cross-domain (locked, n = 112,930, base rate 0.0270)

| reference | AUC | AP | P@10 | P@50 | lift@10 |
|---|---|---|---|---|---|
| unweighted train | 0.5447 | 0.0350 | 0.60 | 0.36 | 22.2× |
| density-ratio reweighted | 0.5525 | 0.0373 | **0.70** | 0.38 | **25.9×** |
| deliberately mismatched | 0.6573 | 0.0604 | 0.30 | 0.22 | 11.1× |

**Domain matching helps where it matters.** Reweighting raises P@10 from 0.60 to
0.70 and lift@10 from 22.2× to 25.9× — the top of the review queue, which is what
an operator actually reads.

**The mismatched reference has higher AUC, and that is a trap.** A 2-class
(SNIa + AGN) prior flags far more objects as unusual, broadening recall across
the whole queue and inflating AUC to 0.6573. But its top-of-queue precision is
less than half (P@10 0.30 vs 0.70). AUC measures whole-queue ordering; the
operationally meaningful quantity is precision at the review budget. **Never
quote AUC without its base rate.**

### 4.3 Domain health

The monitor returns **`abstain`** on the full-scale stream: mean PSI 0.4709,
worst PSI 0.8204 against a fail threshold of 0.25. The worst covariates are
`lc_peak_mag` (PSI 0.82), `phys_distmod` / `phys_z` (0.74) and `lc_snr_max`
(0.63). The stream is fainter, more distant and noisier than the training
reference. The system is correctly reporting that it is outside its validated
domain — and that report is the honest context for every full-scale number above.

## 5. The central negative result: CaRT is not detectable

**CaRT (class 993) scores enrichment 0.0 and recall@200 of 0.000** at full scale.
It is **2,087 of the 3,052 held-out objects — 68% of everything the system is
supposed to find.** This single class is why full-scale AUC sits near chance.

This is not a defect in the ranker. CaRT's median observation is:

| class | median detections | median SNR | median bands |
|---|---|---|---|
| **CaRT (held out)** | **4** | **7.57** | **3** |
| SNII (known) | 4 | 7.57 | 3 |
| SNIa (known) | 5 | 8.72 | 3 |

**Statistically identical to a core-collapse supernova.** Nor is the quality gate
responsible: the gate would abstain CaRT 75.2% of the time, but it abstains SNIa
67.9% and SNII 68.7% — essentially the same rate. With four detections, a
calcium-rich fast transient is not separable from a poorly sampled supernova in
this feature space. Reported as measured.

### Where the data *does* support separation

| population | n | enrichment | recall@200 |
|---|---|---|---|
| muLens-Single (6) | 245 | **71.4×** | 0.127 |
| muLens-Binary (991) | 106 | **58.6×** | 0.104 |
| SLSN-I (95) | 2,786 | 5.7× | 0.010 |
| PISN (994) | 224 | 2.5× | 0.004 |
| ILOT (992) | 363 | 1.6× | 0.003 |
| **CaRT (993)** | **2,087** | **0.0×** | **0.000** |
| KN (64) | 27 | 0.0× | 0.000 |

31 of 35 muLens-Single candidates in the scored stream reach the top 200. Where
the transient is genuinely unlike anything in the prior *and* is sampled well
enough to see, the engine finds it.

On the matched split the same pattern holds: muLens-Single recall@200 0.473
(3.93×), SNIax 0.386, SNIbc 0.357, SNIa-91bg 0.300. KN manages only 0.061
(0.50×) — consistent with v1's finding that KN is the hardest held-out class.

## 6. Weight selection, and why the ablation must not be acted on

Selection searched **216 combinations on the validation split alone**:
validation AP **0.2236** vs `baseline_v1` **0.1458**. Selected weights:
`taxonomy_gap 1.0, simplex_novelty 0.4, cc_weighted 0.35, neighbor_entropy 0.1`.

**Measured optimism** (5 seeds, disjoint select/report halves): select AP 0.2473,
report AP 0.1996, **optimism 0.0477 ± 0.0214**, 0 of 5 seeds negative. The search
flatters itself by about 0.05 AP — real, but bounded, and now quantified rather
than assumed.

### The ablation trap

Ablation on the **locked test split** shows:

| configuration | AUC | AP |
|---|---|---|
| full (selected) | 0.8156 | 0.5399 |
| **with `family_misfit` (+0.2)** | **0.8652** | **0.5867** |
| **with `anomaly_score` (+0.2)** | 0.8349 | 0.5721 |
| without `cc_weighted` | 0.7529 | 0.4434 |
| `taxonomy_gap` only | 0.7676 | 0.4096 |

Two things follow. First, **`cc_weighted` is doing real work**: removing it costs
0.097 AP, the largest single-channel loss, which validates the selection.

Second, **adding `family_misfit` looks better on the test set — and we are not
doing it.** Those numbers come from the locked test split, so promoting the
channel on their basis would be tuning on the held-out data: exactly the leakage
`LeakageGuard` exists to prevent. Both channels were also falsified in v1
(`family_misfit` 0.449, `anomaly_score` 0.467 — at or below chance) and neither
won on validation. **This is recorded as a hypothesis for the next data release,
not a change.** A test-set ablation that tempts you into editing the model is the
mechanism by which benchmarks quietly become fiction.

## 7. Channel power (locked test, measured but not used for tuning)

| channel | AUC | AP |
|---|---|---|
| `cc_weighted` | 0.8322 | 0.5931 |
| `prior_entropy` | 0.7750 | 0.4908 |
| `taxonomy_gap` | 0.7676 | 0.4096 |
| `novelty_gap` | 0.7602 | 0.3426 |
| `family_misfit` | 0.7365 | 0.2488 |
| `anomaly_score` | 0.7071 | 0.2673 |
| `neighbor_entropy` | 0.7048 | 0.1916 |
| `physics_gap` | 0.6764 | 0.2311 |
| `simplex_novelty` | 0.6361 | 0.2286 |

The generic unsupervised detectors remain weak, consistent with v1. They are
retained as zero-weight audit channels — measured, reported, never deleted.

## 8. Base-rate stress

Precision collapses as the prior falls toward the deployment regime. On the
full-scale queue:

| assumed prior | P@100 | lift@100 | R@200 |
|---|---|---|---|
| 1% | 0.529 | 52.9× | 0.0144 |
| 0.1% | 0.919 | 918.9× | 0.0144 |
| 0.01% | 0.991 | 9,912.6× | 0.0144 |

Recall is prior-invariant (0.0144 throughout) — as it must be. **The enormous
lift figures at low priors are an arithmetic consequence of a small denominator,
not evidence of skill.** An enriched-stream precision is never a deployment
number.

### Review budget (matched split)

| budget | novel recovered | precision | lift |
|---|---|---|---|
| 10 | 10.0 | 1.000 | 250× |
| 50 | 40.0 | 0.992 | 248× |
| 100 | 50.0 | 0.968 | 242× |
| 500 | 90.0 | 0.868 | 217× |

At a 10-object review budget the queue is perfect on this split. This is the
number an observing programme should plan against.

## 9. Quality gating and abstention

Gating is **multiplicative and suppress-only** — bad data can never raise a
novelty score. Abstention: 55.5% on the matched split (920/1,659), 31.4%
full-scale (35,515/112,930).

**False-positive audit at k = 100 on the full scored stream** (112,930 objects,
`fullscale_naive`): **74 of the top 100 are not novel**, yet **0 are low
quality** (mean quality 0.9982, minimum 0.90, none below the 0.30 abstain
floor). Mean uncertainty 0.2979. Tiers: 4 critical, 84 high, 12 moderate. 19 of
the 100 would have been abstained.

The same audit recomputed on the 400-object review queue actually shipped to the
dashboard (`data/artefacts/candidates_fullscale.parquet`, ordered by
`novelty_score`) is *not* the same population and gives different numbers:

| depth | novel | false positives | precision | mean quality | min quality | quality < 0.30 |
|---|---|---|---|---|---|---|
| top 10 | 7 | 3 | 0.700 | 0.9963 | 0.9625 | 0 |
| top 50 | 19 | 31 | 0.380 | 0.9952 | 0.8961 | 0 |
| top 100 | 29 | 71 | 0.290 | 0.9956 | 0.8961 | **0** |
| top 200 | 44 | 156 | 0.220 | 0.9968 | 0.8639 | 0 |

The two agree where it matters: in both, **not one false positive is low
quality**. The gate is not leaking bad data into the queue. The misses are
genuine astrophysical confusion — overwhelmingly CaRT — not photometric failure.
That distinction is the whole point of separating ranking quality from data
quality, and it is why §5 rather than §9 holds the explanation for the misses.

Queue-wide the tiers are 4 critical, 84 high, 4,676 moderate, 108,166 routine,
with 35,515 of 112,930 objects abstained (31.4%).

### 9.1 Artifact-injection safety suite (measured)

`scripts/05_artifact_safety.py` injects six controlled defect types into real
light curves and re-scores the whole queue with each corrupted object in place.
The gate's own bar is zero promotions; it does not meet it, and that is reported
as measured rather than tuned away.

| artifact | cases | score increased | climbed the queue | mean Δquality | mean rank gain |
|---|---|---|---|---|---|
| `negative_flux_heavy` | 6 | 1 | 0 | **-0.506** | **-10.7** |
| `error_bar_corruption` | 6 | 1 | 0 | **-0.506** | **-12.2** |
| `duplicate_timestamps` | 6 | 2 | 0 | **-0.506** | **-12.2** |
| `incomplete_band_coverage` | 6 | 3 | 2 | -0.506 | -3.5 |
| `long_cadence_gap` | 6 | 4 | 4 | -0.198 | -1.5 |
| **`single_epoch_spike`** | 6 | **5** | **3** | **0.000** | **+1.3** |
| **total** | **36** | **16** | **9** | -0.370 | |

Two things are true at once. The gate works for most defects: five of six types
drop quality by ~0.51 and push the object **down** the queue (mean rank gain
between -3.5 and -12.2). Only 9 of 36 corrupted objects climbed at all.

**But `single_epoch_spike` is a genuine blind spot.** A single epoch inflated
60x changes quality by **exactly 0.000**, because every gate input is a median or
a count: `q_det_snr_median`, `q_n_det`, `q_n_bands_detected`, `q_err_ratio_median`
are all untouched by one absurd point. `q_err_ratio_max` actually moves the
*wrong* way, since inflating flux lowers `flux_err/|flux|`. 5 of 6 spiked objects
score higher and 3 climb. Fixing this needs a new robust-outlier feature in the
featuriser, which changes every benchmark number - so it is listed in section 12
rather than quietly patched, because the chunks needed to re-run the full-scale
evaluation are no longer present in this workspace.

### 9.2 Early detection by truncation window (measured, negative)

`scripts/04_early_detection.py` truncates each light curve to a window and scores
it with the **full-length model, unrefit** - the realistic deployment case, where
no early-trained model exists.

| window | n | novel | AUC | AP | P@50 | R@200 |
|---|---|---|---|---|---|---|
| 15 d | 2,106 | 196 | **0.437** | 0.085 | 0.120 | 0.087 |
| 30 d | 2,106 | 196 | 0.459 | 0.090 | 0.100 | 0.112 |
| 60 d | 2,106 | 196 | 0.517 | 0.100 | 0.100 | 0.128 |
| 120 d | 2,106 | 196 | 0.522 | 0.099 | 0.040 | 0.122 |
| full | 2,106 | 196 | 0.558 | 0.100 | 0.060 | 0.071 |

**At 15 days the ranking is worse than chance (AUC 0.437 < 0.5).** It is not
merely weak early - it is anti-correlated, and it only crosses 0.5 at ~60 days.
The mechanism is consistent with section 5: the quality gate penalises thin
coverage, and early light curves are thin, so genuinely novel objects that simply
have not been observed much yet are suppressed below the population.

These figures are **not** comparable to the frozen benchmarks in section 4. They
were produced on the 2,106-object chunk-01 working set at a 9.3% base rate, not on
the locked splits, and they are reported here because the run exists - not as a
headline number. The v1 lead-time figures (15 d 0.582 -> full 0.759) were measured
on a different population at a different base rate and cannot be compared to these.

A model trained *on* truncated light curves would do better. That is the P1
early-warning model, explicitly out of scope for this build.

### 9.3 Adversarial domain validation - is it novelty, or just domain shift?

The most dangerous failure mode for a novelty detector is flagging objects
because they come from a different *population* than the reference, not because
they are astrophysically unusual. This is precisely the trap that decided SETI
Breakthrough Listen: the winning team found their models were reading the
background rather than the signal.

The test trains a classifier to distinguish reference from target objects using
the same 401 features the engine uses. On the locked `test_matched` split
(4,621 reference vs 1,659 target, 5-fold):

**Separability AUC = 0.5438.** The populations are effectively
indistinguishable. Within the matched split, "novel" cannot be a stand-in for
"different domain" - the engine is measuring what it claims to measure. Top
discriminating features are shape statistics (`u_t_centroid`, `g_curvature`,
`y_flux_kurt`), not the quality or distance columns that would signal a survey
difference.

**The novelty-vs-domain correlation is NOT established.** It was computed on only
400 of 1,659 target objects - the persisted top-400 review queue, which is
selection-biased by construction. The resulting r=+0.579 must not be read as
evidence that the ranking is domain-driven. Establishing it needs novelty scores
for all 1,659 objects, i.e. a re-run of `03_evaluate.py`.

This matters more on the full-scale stream, where the domain monitor already
returns `abstain` (mean PSI 0.47). The matched-split result does not clear that
stream; it clears the benchmark.

## 10. Uncertainty: a negative result

**Uncertainty is anti-correlated with novelty** at full scale:
mean uncertainty 0.2343 for known objects vs **0.1838 for novel ones**, and
uncertainty-vs-novelty **AUC 0.4219** (below chance).

The bootstrap ensemble is *more* confident about objects the prior has never
seen. That is the expected failure mode of an ensemble trained only on known
classes: a genuinely novel object can sit in a sparse but internally consistent
region and receive confident, wrong predictions. **Uncertainty must not be used
as a novelty signal**, and in this build it is not — it feeds abstention only.

## 11. Reproduction

```bash
python scripts/00_download.py && python scripts/01_featurise.py --all
python scripts/02_train.py && python scripts/03_evaluate.py
python -m pytest tests/ -q      # 228 passed
python scripts/04_early_detection.py && python scripts/05_artifact_safety.py
```

`data/processed/train_features.parquet` is bundled, so all 228 tests run and none
skip. Four of them cover the leakage-guard arming discussed in section 2; they
`skipif` the parquet is absent, so a checkout that deletes it degrades to 224
passed / 4 skipped rather than failing.

## 12. What would actually improve this

1. **More detections per object.** The CaRT failure is a sampling problem. No
   ranking change fixes four data points.
2. **A reference drawn from the target survey.** The domain monitor's `abstain`
   says the current reference is out of domain; reweighting recovered part of the
   gap but cannot manufacture coverage that isn't there.
3. **Add a robust-outlier quality feature.** `single_epoch_spike` is the one
   measured defect the gate cannot see at all (section 9.1). A MAD-based outlier
   ratio, or the share of total flux carried by the single brightest epoch, would
   catch it. This requires a new featuriser column and therefore invalidates
   every number in this report, so it should be done as a deliberate re-run
   rather than a patch.
4. **Test the `family_misfit` hypothesis on fresh data.** It is the single most
   promising open lead in this report, and it is currently un-actionable without
   leaking.

### The featuriser is not bit-reproducible (measured)

Re-featurising the identical train split twice in one process, at the documented
seed, does not reproduce the matrix. **47 of 401 numeric columns differ**, the
largest by 0.0227 (`i_dflux_max`). The affected columns are per-band
detection-only statistics (`*_dflux_max`, `*_dflux_median`, `*_dflux_mean`,
`*_dpeak_time`) plus downstream uncertainty quantiles.

The differences are at float32 last-bit scale - the two largest, 0.0078125 and
0.00390625, are exactly 2^-7 and 2^-8 - so this is accumulation-order noise in
the grouped reductions, not a missing seed. The MC redshift draws *are* seeded
(`seed=42`).

The effect on results is small but not zero: refitting the known-physics prior on
a freshly featurised matrix gave CV accuracy **0.8210** and macro AUC **0.9664**,
versus **0.8258** and **0.9668** in `reports/metrics.json` - a drift of 0.0048 in
accuracy on the same frozen 4,621-object `prior_fit` split.

Consequences for anyone reading the numbers:

- "seed 42" fixes the *splits, the weight search and the prior's own randomness*.
  It does not make the feature matrix bit-identical.
- Benchmark figures are stable to roughly the third decimal, not exactly
  repeatable. Treat +-0.005 on accuracy-class metrics as the reproducibility
  floor until the grouped reductions are pinned (e.g. by computing them in
  float64 and rounding once at the end).
- `reports/metrics.json` is the authoritative record of what was measured. A
  re-run will land close but not on top of it.

See `CHANGELOG.md` for the 24 genuine bugs found and fixed during this rebuild, including the four that made the artifact-safety suite unrunnable and the one that made its result meaningless.
