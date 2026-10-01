"""Broker adapters, replay isolation, uncertainty and the explanation layer."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cne.adapters import (
    ADAPTERS,
    AdapterError,
    ReplayRunner,
    RubinLikeAdapter,
    detect_adapter,
    write_sample_alerts,
)
from cne.schema import AlertPacket
from cne.uncertainty import (
    AbstentionPolicy,
    BootstrapEnsemble,
    evaluate_uncertainty_value,
    feature_reliability,
)
from cne.explain import analogue_block, top_feature_deviations


def _packet(oid=1, n=30, survey="ztf-like"):
    rng = np.random.default_rng(oid)
    return AlertPacket(
        object_id=oid,
        survey=survey,
        mjd=(60000 + np.arange(n) * 2.0).astype("float32"),
        passband=(np.arange(n) % 6).astype("int8"),
        flux=(100 * np.exp(-np.arange(n) / 10.0) + rng.normal(0, 2, n)).astype("float32"),
        flux_err=np.full(n, 2.0, dtype="float32"),
        detected=np.ones(n, dtype="int8"),
        context={"ra": 150.0, "decl": 2.0, "hostgal_photoz": 0.3, "hostgal_photoz_err": 0.03},
        quality={"realbogus": 0.95},
    )


class TestPacketValidation:
    def test_a_clean_packet_has_no_problems(self):
        assert _packet().validate() == []

    def test_empty_packet_is_rejected(self):
        packet = _packet()
        packet.mjd = np.array([], dtype="float32")
        assert "empty light curve" in packet.validate()

    def test_non_positive_error_is_rejected(self):
        packet = _packet()
        packet.flux_err[3] = 0.0
        assert any("flux_err" in p for p in packet.validate())

    def test_out_of_range_passband_is_rejected(self):
        packet = _packet()
        packet.passband[0] = 9
        assert any("passband" in p for p in packet.validate())

    def test_round_trips_through_a_frame(self):
        packet = _packet()
        frame = packet.to_frame()
        assert list(frame.columns) == ["object_id", "mjd", "passband", "flux", "flux_err", "detected_bool"]
        assert len(frame) == packet.n_points


class TestAdapters:
    def test_ztf_packet_is_recognised(self):
        assert detect_adapter({"candid": 1, "candidate": {}}).survey == "ztf-like"

    def test_rubin_packet_is_recognised(self):
        assert detect_adapter({"alertId": 1, "diaSource": {}}).survey == "rubin-like"

    def test_unknown_shape_raises(self):
        with pytest.raises(AdapterError):
            detect_adapter({"foo": "bar"})

    def test_ztf_julian_date_is_converted_to_mjd(self):
        raw = {
            "candid": 42, "objectId": "ZTF42",
            "candidate": {"candid": 42, "ra": 10.0, "dec": 20.0, "jd": 2460000.5,
                          "fid": 2, "flux": 120.0, "fluxerr": 3.0, "isdiffpos": 1, "drb": 0.9},
            "prv_candidates": [{"jd": 2459998.5, "fid": 1, "flux": 80.0,
                                "fluxerr": 2.5, "isdiffpos": 1}],
        }
        packet = ADAPTERS["ztf-like"].to_packet(raw)
        assert packet.n_points == 2
        # JD 2460000.5 - 2400000.5 = MJD 60000
        assert float(packet.mjd.max()) == pytest.approx(60000.0)
        assert float(packet.mjd.min()) == pytest.approx(59998.0)
        assert float(packet.quality["realbogus"]) == 0.9
        assert set(packet.passband.tolist()) <= {1, 2}

    def test_rubin_band_letters_map_to_passbands(self):
        raw = {
            "alertId": 7, "diaSource": {"diaSourceId": 7, "midPointTai": 60000.0, "band": "z",
                                        "psfFlux": 90.0, "psfFluxErr": 2.0, "ra": 5.0, "decl": -5.0},
            "diaSourceHistory": [{"midPointTai": 59998.0, "band": "g", "psfFlux": 60.0, "psfFluxErr": 2.0}],
            "diaObject": {"diaObjectId": 7, "zphot": 0.4, "zphotErr": 0.05},
        }
        packet = ADAPTERS["rubin-like"].to_packet(raw)
        assert set(packet.passband.tolist()) == {1, 4}
        assert packet.context["hostgal_photoz"] == 0.4

    def test_a_packet_with_no_usable_observations_raises(self):
        with pytest.raises(AdapterError):
            ADAPTERS["ztf-like"].to_packet({"candid": 1, "candidate": {"ra": 0.0}, "prv_candidates": []})

    def test_provenance_is_carried(self):
        packet = ADAPTERS["ztf-like"].to_packet({
            "candid": 5, "objectId": "ZTF5",
            "candidate": {"jd": 2460000.5, "fid": 2, "flux": 10.0, "fluxerr": 1.0},
        })
        assert packet.provenance["survey"] == "ztf-like"
        assert "adapter_version" in packet.provenance

    def test_sample_alert_file_round_trips(self, tmp_path):
        packets = [_packet(1), _packet(2, n=18)]
        path = write_sample_alerts(tmp_path / "alerts.jsonl", packets, survey="ztf-like")
        runner = ReplayRunner(lambda p: {"object_id": p.object_id, "n": p.n_points})
        results, report = runner.run(path)
        assert report.n_failed == 0
        assert report.n_succeeded == 2
        # ZTF objectId is an opaque string, so the adapter hashes it rather than
        # recovering the integer we put in; what matters is that distinct alerts
        # stay distinct and every packet survives the round trip.
        assert len({r["object_id"] for r in results}) == 2


class TestReplay:
    def test_one_bad_packet_does_not_stop_the_stream(self, tmp_path):
        lines = [
            json.dumps({"candid": 1, "candidate": {"jd": 2460000.5, "fid": 2, "flux": 10.0, "fluxerr": 1.0}}),
            "{not json at all",
            json.dumps({"candid": 3, "candidate": {"jd": 2460001.5, "fid": 1, "flux": 20.0, "fluxerr": 1.0}}),
        ]
        path = tmp_path / "mixed.jsonl"
        path.write_text("\n".join(lines))
        runner = ReplayRunner(lambda p: {"object_id": p.object_id})
        results, report = runner.run(path)
        assert report.n_packets == 3
        assert report.n_succeeded == 2
        assert report.n_failed == 1
        assert len(results) == 2

    def test_throughput_and_latency_are_reported(self, tmp_path):
        path = write_sample_alerts(tmp_path / "a.jsonl", [_packet(i) for i in range(5)])
        _results, report = ReplayRunner(lambda p: {"object_id": p.object_id}).run(path)
        assert report.throughput_per_second > 0
        assert report.mean_latency_ms >= 0

    def test_limit_is_respected(self, tmp_path):
        path = write_sample_alerts(tmp_path / "a.jsonl", [_packet(i) for i in range(10)])
        _results, report = ReplayRunner(lambda p: {"object_id": p.object_id}).run(path, limit=3)
        assert report.n_packets == 3


class TestUncertainty:
    @pytest.fixture(scope="class")
    def ensemble(self, synthetic):
        from cne.taxonomy import KNOWN_CLASS_CODES

        feats, labels = synthetic["features"], synthetic["labels"]
        astro = [c for c in feats.columns if c not in ("object_id",) and not c.startswith("q_")]
        known = np.isin(labels, sorted(KNOWN_CLASS_CODES))
        return BootstrapEnsemble(n_models=3, seed=42, n_estimators=40).fit(
            feats[known][astro], labels[known], astro)

    def test_uncertainty_is_bounded(self, ensemble, synthetic):
        astro = ensemble.feature_names_
        unc, max_prob, disagreement = ensemble.uncertainty(synthetic["features"][astro])
        assert np.all(unc >= 0) and np.all(unc <= 1)
        assert np.all(max_prob > 0) and np.all(max_prob <= 1)
        assert np.all(disagreement >= 0) and np.all(disagreement <= 1)

    def test_ensemble_is_deterministic(self, ensemble, synthetic):
        astro = ensemble.feature_names_
        a = ensemble.uncertainty(synthetic["features"][astro])[0]
        b = ensemble.uncertainty(synthetic["features"][astro])[0]
        np.testing.assert_allclose(a, b)

    def test_missing_reliability_column_defaults_to_one(self):
        out = feature_reliability(pd.DataFrame({"object_id": [1, 2]}))
        np.testing.assert_array_equal(out, np.ones(2))

    def test_abstention_reasons_are_specific(self):
        policy = AbstentionPolicy()
        assert policy.decide(0.1, 0.2, 1.0).reason == "low data quality"
        assert "uncertainty" in policy.decide(1.0, 0.99, 1.0).reason
        assert policy.decide(1.0, 0.2, 0.0).reason == "no reliable distance information"
        assert "domain" in policy.decide(1.0, 0.2, 1.0, domain_status="abstain").reason
        assert policy.decide(1.0, 0.2, 1.0).abstain is False

    def test_degraded_domain_does_not_abstain_but_is_flagged(self):
        decision = AbstentionPolicy().decide(1.0, 0.2, 1.0, domain_status="degraded")
        assert decision.abstain is False
        assert "degraded" in decision.reason

    def test_uncertainty_value_is_measured_on_the_queue(self):
        rng = np.random.default_rng(0)
        y = (rng.random(500) < 0.2).astype(int)
        scores = np.where(y == 1, rng.uniform(0.4, 1.0, 500), rng.uniform(0.0, 0.9, 500))
        unc = rng.uniform(0, 1, 500)
        out = evaluate_uncertainty_value(y, scores, unc)
        assert "all_objects" in out
        assert any(k.startswith("keep_most_confident") for k in out)


class TestExplanation:
    def test_an_unreliable_neighbourhood_is_said_so(self):
        block = analogue_block([{"class_code": 90, "distance": 30.0}])
        assert block["reliable"] is False
        assert "no trustworthy" in block["reason"]

    def test_a_pure_neighbourhood_reports_its_dominant_class(self):
        analogs = [{"class_code": 90, "distance": 1.0}] * 4 + [{"class_code": 42, "distance": 1.5}]
        block = analogue_block(analogs)
        assert block["reliable"] is True
        assert block["dominant_class"] == "SNIa"
        assert block["class_purity"] == pytest.approx(0.8)

    def test_deviations_need_a_class_reference(self):
        row = pd.Series({"phys_absmag_r": -12.0})

        class _Stub:
            prior_ = None

        assert top_feature_deviations(row, _Stub(), 90) == []
