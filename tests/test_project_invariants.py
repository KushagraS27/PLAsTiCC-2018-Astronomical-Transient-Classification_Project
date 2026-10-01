"""Project invariants: taxonomy, config integrity, and bounded scientific language."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from cne.config import CNEConfig, load_config
from cne.taxonomy import (
    KNOWN_CLASS_CODES,
    PASSBAND_NAMES,
    POPULATIONS,
    WITHHELD_FROM_PRIOR,
    describe,
    family_of,
    is_held_out,
    name_of,
)
from cne.version import FROZEN_V1_BASELINE, __version__, library_versions, stamp

ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_PHRASES = ("NEW DISCOVERY CONFIRMED", "CONFIRMED NEW PHYSICS", "PROVES NEW PHYSICS")


class TestTaxonomy:
    def test_class_count_matches_the_published_plasticc_note(self):
        """Zenodo 2539456 note2 lists 14 labelled + 5 injected populations."""
        assert len(POPULATIONS) == 19

    def test_known_classes_number_twelve(self):
        assert len(KNOWN_CLASS_CODES) == 12

    def test_held_out_populations_are_never_in_the_prior(self):
        assert KNOWN_CLASS_CODES.isdisjoint(WITHHELD_FROM_PRIOR)

    def test_every_population_has_a_family(self):
        for code in POPULATIONS:
            assert family_of(code) in ("galactic", "extragalactic")

    def test_novel_populations_are_marked(self):
        for code in (64, 991, 992, 993, 994):
            assert is_held_out(code), f"class {code} should be held out"

    def test_name_lookup_is_total(self):
        for code in list(POPULATIONS) + [99999]:
            assert isinstance(name_of(code), str) and name_of(code)

    def test_describe_is_serialisable(self):
        rows = describe()
        assert len(rows) == len(POPULATIONS)
        assert {"code", "name", "family", "in_known_prior", "held_out"} <= set(rows[0])

    def test_passband_mapping_covers_ugrizy(self):
        assert set(PASSBAND_NAMES.values()) == {"u", "g", "r", "i", "z", "y"}


class TestConfig:
    def test_default_config_loads(self):
        cfg = load_config(ROOT / "configs" / "default.yaml")
        assert cfg.seed == 42
        assert cfg.weights["taxonomy_gap"] == 1.0

    def test_every_configured_channel_is_a_known_channel(self):
        from cne.novelty import ALL_CHANNELS

        cfg = load_config(ROOT / "configs" / "default.yaml")
        assert set(cfg.weights) == set(ALL_CHANNELS)

    def test_unknown_config_key_is_rejected_not_ignored(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("meta:\n  seed: 42\nprior:\n  n_estimatorz: 5\n")
        with pytest.raises(KeyError):
            CNEConfig.load(bad)

    def test_overrides_merge_deeply(self):
        cfg = load_config(ROOT / "configs" / "default.yaml", overrides={"prior": {"n_folds": 7}})
        assert cfg.prior.n_folds == 7
        assert cfg.prior.n_estimators > 0  # untouched keys survive

    def test_fingerprint_is_stable_and_sensitive(self):
        cfg = load_config(ROOT / "configs" / "default.yaml")
        assert cfg.fingerprint() == cfg.fingerprint()
        other = load_config(ROOT / "configs" / "default.yaml", overrides={"prior": {"n_folds": 3}})
        assert cfg.fingerprint() != other.fingerprint()

    def test_quality_is_enabled_and_suppress_only(self):
        cfg = load_config(ROOT / "configs" / "default.yaml")
        assert cfg.quality.enabled is True
        assert 0 < cfg.quality.exponent <= 1.0

    def test_v2_flags_are_declared(self):
        cfg = load_config(ROOT / "configs" / "default.yaml")
        for flag in ("locked_test_manifest", "nested_weight_selection", "domain_matched_reference",
                     "base_rate_stress", "artifact_safety_suite", "uncertainty_propagation",
                     "abstention", "domain_shift_monitor", "replay_adapter"):
            assert hasattr(cfg.v2, flag)


class TestScientificLanguage:
    """Blueprint rule: the phrase 'NEW DISCOVERY CONFIRMED' is forbidden."""

    def _source_files(self):
        patterns = ("cne/**/*.py", "api/**/*.py", "scripts/*.py", "web/src/**/*.jsx",
                    "web/src/**/*.js", "configs/*.yaml", "reports/*.md")
        files = []
        for pattern in patterns:
            files.extend(ROOT.glob(pattern))
        return sorted({f for f in files if f.is_file()})

    def test_no_forbidden_phrase_in_the_codebase(self):
        offenders = []
        for path in self._source_files():
            text = path.read_text(errors="ignore").upper()
            for phrase in FORBIDDEN_PHRASES:
                # The rule is about output text, not about the rule's own statement.
                for match in re.finditer(re.escape(phrase), text):
                    # Statements of the rule itself are legitimate; look at both
                    # sides, since "is forbidden" often follows the phrase.
                    context = text[max(0, match.start() - 80): match.end() + 80]
                    if any(w in context for w in ("FORBIDDEN", "NEVER", "BANNED",
                                                  "MUST NOT", "DO NOT EMIT", "RULE")):
                        continue
                    offenders.append(f"{path.relative_to(ROOT)}: {phrase}")
        assert not offenders, offenders

    def test_the_rule_itself_is_documented(self):
        assert "NEW DISCOVERY CONFIRMED" in (ROOT / "cne" / "__init__.py").read_text()

    def test_no_claim_of_confirmed_discovery_in_generated_explanations(self):
        from cne.explain import _summary

        summary = _summary({"best_fit_prob": 0.4}, 90, "taxonomy_gap", [], {"reliable": False})
        upper = summary.upper()
        assert "CONFIRMED DISCOVERY" not in upper
        assert "CANDIDATE" in upper or "FOLLOW-UP" in upper


class TestProvenance:
    def test_version_and_seed_are_stamped(self):
        payload = stamp()
        assert payload["cne_version"] == __version__
        assert payload["seed"] == 42
        assert "numpy" in payload["libraries"]

    def test_frozen_baseline_is_recorded(self):
        assert FROZEN_V1_BASELINE["ai05_roc_auc"] == pytest.approx(0.7291)
        assert FROZEN_V1_BASELINE["domain_matched_prior_roc_auc"] == pytest.approx(0.818)
        assert FROZEN_V1_BASELINE["out_of_domain_prior_roc_auc"] == pytest.approx(0.610)

    def test_library_versions_are_complete(self):
        versions = library_versions()
        for name in ("python", "numpy", "pandas", "lightgbm"):
            assert name in versions
