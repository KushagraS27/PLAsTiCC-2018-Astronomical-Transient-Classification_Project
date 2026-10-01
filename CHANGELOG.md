# Changelog — Cosmic Novelty Engine v2.0.0

Codename **Aediles**. Rebuild of CNE v1.0.0 plus the P0 upgrade set. Every number
in `reports/metrics.json` carries dataset version, split manifest, seed, config id
and base rate.

The phrase `NEW DISCOVERY CONFIRMED` is forbidden in this codebase and is enforced
by a test (`tests/test_project_invariants.py`). Permitted wording only: *candidate
anomaly*, *potentially novel candidate*, *requires expert follow-up*, *poorly
explained by current known populations*.

---

## Added

- **Locked test manifest with write-once semantics** (`cne/manifests.py`).
  `reports/manifests/splits_v2.json` freezes `prior_fit` 4,621 / `validation` 1,568
  / `test_matched` 1,659 / `test_fullscale` 112,930 at seed 42. The file is
  `chmod 0444`; `verify()` **recomputes** the fingerprints from the ids on disk
  rather than comparing the recorded block to itself, so hand-edits are detected.
- **LeakageGuard** — records every object id each tuning stage touches and raises
  if a locked-test id appears. Weight search reads only `validation`.
- **Nested weight selection** on the validation split alone (216 combinations).
- **Base-rate stress** (`cne/stress.py`) — precision/AP/lift reweighted to
  10%/1%/0.1%/0.01% priors, so an enriched-stream number is never mistaken for a
  deployment number.
- **Artifact-injection safety suite** (`cne/artifacts.py`) — six corruption types
  pushed through the production scoring path; the only passing outcome is zero
  promotions.
- **Uncertainty propagation and abstention** (`cne/uncertainty.py`).
- **Broker replay adapters** (`cne/adapters.py`) — ZTF and Rubin alert packets,
  with per-packet provenance and a crash-isolating replay runner.
- **Domain-shift monitor** (`cne/domain.py`) — PSI/KS per covariate; emits no
  channel a ranker can consume, so drift can never be read as novelty.
- **Test suite**: 224 tests across 9 files.
- **FastAPI service + React dashboard** — all ten v1 endpoints preserved.

## Fixed — genuine bugs the new tests and the full-scale run exposed

Each of these was a real defect, not a test-expectation problem.

1. **`mc_redshift_draws` depended on batch size.** `standard_normal((n_draws,
   n_objects))` gave object *i* different uncertainty features depending on how
   many objects were scored alongside it. Replaced with stratified fixed normal
   quantiles `norm.ppf((k+0.5)/n)`.
2. **`comoving_distance_mpc` built its integration grid from `z.max()` of the
   batch.** Same class of bug. Replaced with a fixed 1024-node trapezoid grid in
   `u = ln(1+z)` plus `np.interp`; accurate to <0.1% against reference values.
3. **Rank-normalised channels were batch-dependent.** `_rank01` normalised inside
   the incoming batch, so scoring in chunks to cap memory changed the answer by up
   to **0.26**. Added `QuantileMapper`: grids are frozen on the reference at fit
   time. Residual is now bounded by one grid step (1/N_ref) and comes only from
   float reduction order in the autoencoder (2.8e-14).
4. **`weighted_roc_auc` counted negatives *above* each positive.** Returned
   0.1201 where the correct value was 0.8799 — exactly `1 − AUC`. Rewritten as a
   block-sweep weighted Mann-Whitney; now matches sklearn exactly under uniform
   weights.
5. **`harvest_stream` was not low-memory despite its docstring.** `del chunk`
   freed the raw input while `pd.concat` accumulated the entire result — ~114M
   rows for PLAsTiCC chunk 02. OOM-killed the harvest. Now writes each block as a
   parquet row group.
6. **`build_stream_features` accumulated every chunk's raw photometry** in
   `lc_parts` before concatenating (~1.7 GB for three chunks). Now streams
   row groups into the light-curve cache.
7. **`_featurise_in_chunks` read the whole chunk** with `pd.read_parquet(part)`.
   Replaced with `_featurise_parquet`, which walks row groups and holds back only
   the boundary object.
8. **`class_weight` broke on rare-class CV folds.** LightGBM resolves
   `class_weight` keys against the classes present in the fold it is given, so a
   fold missing a rare class raised `KeyError: 14`. Replaced with per-row
   `sample_weight`.
9. **Fold predictions could not fill the OOF matrix.** A fold that omitted a rare
   class returned fewer columns than there are global classes. Folds are now
   scattered onto their global class index, in both `fit` and `predict_proba`.
10. **`_fit_weighted` never set `feature_names_` on the wrapped classifier**, so
    `probabilities()` selected zero columns: *"Found array with 0 feature(s)
    (shape=(10, 0))"*.
11. **`roc_auc_score(..., multi_class="ovr")` raised on 2-class fits.** sklearn
    silently routes to binary scoring and rejects the `(n, 2)` matrix. Reachable
    in production: the deliberately out-of-domain reference has two classes. Added
    `_macro_auc`.
12. **`weight_selection_optimism` used `rng.permutation(n) < 0.5`.** `permutation`
    returns a shuffled *integer* array, so only the element equal to zero was
    `True`: SELECT held one object with no positives, every AP was NaN, and the
    study silently reported the `-1.0` sentinel as a finding. Now
    `rng.random(n) < 0.5`, with degenerate halves skipped.
13. **`domain_degradation_study` was called with `fullscale_domain_matched` in
    both** the matched and reweighted slots, making two of three rows
    bit-identical.
14. **`build_explanations` assigned a list-of-dicts through `.loc` with a boolean
    mask**, which pandas broadcasts element-wise: *"Must have equal len keys and
    value"*. Now assigns the whole column at once.
15. **`col_temp_proxy = 1/|g−r|` was unbounded** (reached 851 for near-neutral
    colours). Now `1/(0.2 + |g−r|)`, max 5.0.
16. **`rare_preserving_subsample` overran its budget** (3,002 for
    `max_stream=2000`) because the `rare_cap+1` floor was unconditional.
17. **Artifact injectors used label-based `.loc` with positional indices**, which
    silently corrupted rows on a sliced light curve. Now `iloc`.
18. **Density-ratio weights clipped *before* normalising**, so the documented cap
    did not hold (reached 12.3 at cap 5.0). The cap is now applied in scale-free
    space and is the hard invariant; the mean is reported, not pinned to 1.
19. **`agreement` rank-normalised the spread within the scored batch**, making the
    final score chunk-dependent. Replaced with a coefficient of dispersion.
20. **`run_full_evaluation` read `featuriser_.feature_names()`**, which raises when
    features come from parquet because `transform()` never ran in that process.
21. **`LeakageGuard` was a silent no-op on the path the real run takes.**
    `CNEPipeline.__init__` installs a manifest-less default guard locked on
    `validation`, and `make_manifest()` only rebuilt it on the `force=True` path.
    The cached-manifest path returned early at line 279, so every real evaluation
    ran with `locked_split='validation'` and **`locked_n=0`**. With zero locked ids
    `assert_clean()` could never fire, and the run reported a clean guard while
    checking nothing. Fixed at `cne/pipeline.py:288`: the cached path now runs
    `manifest_.verify(path)` and installs `LeakageGuard(self.manifest_,
    locked_split='test_matched')` before returning. Four regressions added to
    `tests/test_manifests.py` (`TestGuardIsActuallyArmed`) asserting the guard is
    armed, fires when a locked id is touched, stays clean on validation-only
    reads, and refuses a tampered manifest.

    **Note on `reports/metrics.json`:** it was written by the pre-fix run, so its
    `leakage_guard` block still reads `locked_split='validation', locked_n=0`.
    No leakage actually occurred — `weight_search` recorded exactly 1,568 ids,
    equal to `len(validation)`, and `validation ∩ test_matched = 0` — but that is
    established independently of the guard. Re-running `03_evaluate.py` writes an
    armed record. A safety control that reports success without checking is worse
    than no control, so the guard's own `report()['locked_n'] > 0` is now asserted.

22. **`scripts/04` and `scripts/05` called APIs that do not exist.** Both were
    written but never executed, and both crashed immediately:
    `RankingWeights.load(...)` (no such classmethod - the frozen config is a YAML
    mapping under `weights:`), `pipeline.cfg.paths.weights_v2` (`CNEConfig` has no
    `paths`), `stamp(pipeline.cfg)` (`stamp` is `**extra`-only, so a positional arg
    is a `TypeError`), and `fit_engine()` called before `make_manifest()` /
    `build_train_features()` (`'NoneType' object has no attribute 'split'`).
    A script that has never been run is not a deliverable; all four are fixed and
    `05_artifact_safety.py` now runs to completion.
23. **The artifact-safety comparison was invalid.** `score_fn` scored each
    corrupted light curve *on its own*, then compared it to a baseline computed
    from a 2,106-object batch. The evidence channels are rank-normalised against
    the scored batch, so a solo object is not comparable: measured over 40
    objects, solo-vs-batch novelty differed by a **mean of -0.243 (max |delta|
    0.523)** with **0/40 scores matching**. Every corrupted object also reported
    `rank = 1`, because a batch of one is trivially first. The suite was therefore
    measuring batch size, not data quality, and reported 16/36 promotions that
    were mostly an artifact of the mismatch. Fixed: `score_fn` now re-scores the
    whole queue with the one corrupted object in place, so baseline and corrupted
    scores share a batch context and the ranks are real.
24. **`per_artifact` in the safety report was stale.** It was computed inside
    `ArtifactSafetySuite.run()` under the pre-re-expression promotion rule and
    never refreshed, so it claimed `n_promoted=5` for all six artifact types while
    the per-case records it summarises said otherwise. The script's table also
    printed `n_cases` / `mean_quality_drop`, keys the payload does not contain,
    producing `cases=0` and `nan`. Both now read the real fields.

- **`cne/simulator.py`** + `tests/test_simulator.py` (9 tests) - offline synthetic
  rare-transient simulator (CaRT, KN, plus SNII control), the CNE analogue of the
  signal generator that won SETI. Framed honestly: it demonstrates detector
  *capacity* on separable templates but does NOT claim to fix real CaRT, which is
  a measurement/coverage limitation (its median observation is identical to SNII).
- **`reports/SUBMISSION.md`** - the SuperNova submission package required by
  Prompt 13 item 5: problem statement, innovation summary, technical workflow,
  evaluation plan, limitations, and future work.

## Added - SuperNova submission package

- **`cne/adversarial.py`** - adversarial domain validation, adapted from the
  diagnostic that won SETI Breakthrough Listen (Team Watercooled, 1st place).
  Their models were scoring on the *background* rather than the signal, and they
  found it by grouping predictions by train-likeness. The same failure mode is the
  most dangerous one for a novelty detector: an object can look novel because it
  comes from a different population, not because it is astrophysically unusual.
  A reference-vs-target classifier settles it. Measured on the locked
  `test_matched` split (4,621 reference vs 1,659 target, 401 features):
  **AUC 0.5438** - the populations are effectively indistinguishable, so within
  the matched split novelty is *not* a proxy for domain shift. Five tests.
  The novelty-vs-domain correlation is recorded but explicitly **not**
  established: only 400 of 1,659 target objects have persisted novelty scores,
  and that top-400 truncation is selection-biased by construction.
- **`scripts/06_figures.py`** - the eight figures Prompt 13 requires, all drawn
  from `reports/metrics.json` and the frozen artefacts so no number is
  hand-typed: architecture, domain-match comparison, ablation, precision@K,
  base-rate stress, early-detection curve, candidate explanation, false-positive
  audit. Written to `reports/figures/`.

## Corrected test expectations (the code was right)

- Galactic-centre longitude wraps: compare `min(lon, 360 − lon)`.
- "Extragalactic with no distance" must use z > 0; z = 0 is legitimately galactic.
- Hand-computed ROC-AUC for the 8-point fixture is 11/15, not 10/15.
- `single_epoch_spike` scales a **random** row, not the brightest.
- ZTF `jd` 2460000.5 is MJD 60000 (the test had a doubled offset).
- ZTF `objectId` is an opaque string, so the adapter hashes it rather than
  recovering the integer that went in.

## Disclosed limitations

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



- **The 2.70% full-scale base rate is a sampling artifact.** Chunks 02/03 were
  thinned by `rare_preserving_subsample` (40,000 of 345,997 objects each), which
  keeps every rare-class object. Chunk 01, taken whole, has a **0.60%** base rate.
  Use the stress table, not the raw aggregate, for deployment expectations.
- **Class 6 (muLens-Single) has only 151 training examples** and is disclosed as
  weakly represented wherever it appears.
- **KN (class 64) has n = 27** at full scale and carries `low_n_warning`.
- `high_error` and `duplicates` quality penalties are structurally dead on
  PLAsTiCC (0.000 frequency). Retained for real surveys.
- P1 (lead-time-trained early-warning model) and P2 (contrastive/ANN embeddings,
  multimodal) are out of scope; interfaces are reserved only.
