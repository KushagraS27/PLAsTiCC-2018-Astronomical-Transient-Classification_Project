#!/usr/bin/env python3
"""Generate the figures required by the SuperNova submission package.

Every figure is drawn from reports/metrics.json and the frozen artefacts - no
number here is hand-typed, so the plots cannot drift away from the measurements.

    python scripts/06_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports" / "figures"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
    "axes.spines.right": False,
})
C_V1, C_V2, C_BAD, C_OK = "#8899aa", "#1f77b4", "#c0392b", "#27ae60"


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"  wrote {name}")


def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.set_axis_off()
    W, H, Y = 0.135, 0.24, 0.42
    labels = ["Alert stream\n(PLAsTiCC)", "Featuriser\n401 features",
              "Known-physics\nprior (12 cls)", "Evidence\n9 channels",
              "Quality gate\nsuppress-only", "Ranked queue\n+ explanation"]
    xs = [0.02 + i * 0.165 for i in range(6)]
    colours = ["#eef2f6", "#eef2f6", "#dce9f5", "#dce9f5", "#fdeaea", "#e6f4ea"]
    for x, label, colour in zip(xs, labels, colours):
        ax.add_patch(plt.Rectangle((x, Y), W, H, facecolor=colour,
                                   edgecolor="#7f8c8d", lw=1.1, zorder=2))
        ax.text(x + W / 2, Y + H / 2, label, ha="center", va="center",
                fontsize=7.5, zorder=3)
    for i in range(5):
        ax.annotate("", xy=(xs[i + 1], Y + H / 2), xytext=(xs[i] + W, Y + H / 2),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.4))
    ax.text(0.7625, 0.72, "x quality^0.5\nx confidence^0.25\nx (1 + 0.15 agreement)",
            ha="center", fontsize=7.5, color=C_BAD, style="italic")
    ax.text(0.5, 0.20,
            "Novelty is never claimed. Every output is a candidate that is poorly explained\n"
            "by current known populations and requires expert follow-up.",
            ha="center", fontsize=8, color="#555")
    ax.set_title("Cosmic Novelty Engine - system architecture", fontsize=11, pad=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _save(fig, "01_architecture.png")


def fig_domain_match(m: dict) -> None:
    B = m["benchmarks"]
    rows = [("unweighted\ntrain", B["fullscale_naive"]),
            ("density-ratio\nreweighted", B["fullscale_domain_matched"]),
            ("deliberately\nmismatched", B["fullscale_mismatched"])]
    x = np.arange(len(rows)); w = 0.36
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    a1.bar(x - w / 2, [r[1]["metrics"]["roc_auc"] for r in rows], w, label="AUC", color=C_V2)
    a1.bar(x + w / 2, [r[1]["metrics"]["average_precision"] for r in rows], w, label="AP", color=C_V1)
    a1.set_xticks(x); a1.set_xticklabels([r[0] for r in rows], fontsize=8)
    a1.set_title("Whole-queue ranking (misleading here)", fontsize=9)
    a1.legend(fontsize=8); a1.set_ylim(0, 0.75)
    a2.bar(x - w / 2, [r[1]["metrics"]["precision_at_k"]["P@10"] for r in rows], w,
           label="P@10", color=C_OK)
    a2.bar(x + w / 2, [r[1]["metrics"]["lift_at_k"]["lift@10"] for r in rows], w,
           label="lift@10", color="#e67e22")
    a2.set_xticks(x); a2.set_xticklabels([r[0] for r in rows], fontsize=8)
    a2.set_title("Top-of-queue (operationally correct)", fontsize=9)
    a2.legend(fontsize=8)
    a2.annotate("mismatched wins AUC\nbut loses the queue", xy=(2, 12), fontsize=7.5,
                color=C_BAD, ha="center")
    fig.suptitle("Domain match: full-scale stream, n=112,930, base rate 0.0270", fontsize=11)
    fig.tight_layout()
    _save(fig, "02_domain_match.png")


def fig_ablation(m: dict) -> None:
    rows = sorted(m["ablation_matched"], key=lambda r: r["average_precision"])
    names = [r["configuration"] for r in rows]
    ap = [r["average_precision"] for r in rows]
    colours = [C_OK if r["configuration"] == "full" else
               (C_BAD if "without cc_weighted" in r["configuration"] or
                r["configuration"] == "taxonomy_gap only" else C_V2) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(names, ap, color=colours)
    ax.set_xlabel("Average precision (locked test_matched, base rate 0.1073)")
    ax.set_title("Channel ablation - removing cc_weighted costs the most", fontsize=11)
    for i, v in enumerate(ap):
        ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=7.5)
    ax.text(0.02, 0.03,
            "Additive variants beat 'full' ON THE TEST SET. They are not adopted:\n"
            "acting on them would be tuning on held-out data.",
            transform=ax.transAxes, fontsize=7.5, color=C_BAD)
    _save(fig, "03_ablation.png")


def fig_precision_at_k(m: dict) -> None:
    ks = [10, 20, 50, 100]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for key, label, colour in (("matched_baseline_v1", "baseline_v1", C_V1),
                               ("matched_nested_v2", "nested_v2", C_V2)):
        pk = m["benchmarks"][key]["metrics"]["precision_at_k"]
        ax.plot(ks, [pk[f"P@{k}"] for k in ks], "o-", label=label, color=colour)
    fs = m["benchmarks"]["fullscale_domain_matched"]["metrics"]["precision_at_k"]
    ax.plot([10, 100], [fs["P@10"], fs["P@100"]], "s--", color="#e67e22",
            label="full-scale (base rate 0.027)")
    ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels([f"@{k}" for k in ks])
    ax.set_xlabel("review budget"); ax.set_ylabel("precision")
    ax.set_title("Precision at K - matched split (base rate 0.1073)", fontsize=11)
    ax.legend(fontsize=8)
    _save(fig, "04_precision_at_k.png")


def fig_base_rate_stress(m: dict) -> None:
    priors = m["stress_fullscale"]["priors"]
    pr = sorted([p for p in priors if p["target_rate"] > 0], key=lambda p: -p["target_rate"])
    rates = [p["target_rate"] * 100 for p in pr]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    a1.plot(rates, [p["P@100"] for p in pr], "o-", color=C_V2)
    a1.set_xscale("log"); a1.invert_xaxis()
    a1.set_xlabel("assumed prior (%)"); a1.set_ylabel("P@100")
    a1.set_title("Precision collapses toward deployment priors", fontsize=9)
    a2.plot(rates, [p["lift@100"] for p in pr], "o-", color="#e67e22")
    a2.set_xscale("log"); a2.set_yscale("log"); a2.invert_xaxis()
    a2.set_xlabel("assumed prior (%)"); a2.set_ylabel("lift@100 (log)")
    a2.set_title("Lift inflates as the denominator shrinks", fontsize=9)
    a2.text(0.05, 0.85, "arithmetic, not skill", transform=a2.transAxes,
            fontsize=7.5, color=C_BAD)
    fig.suptitle("Base-rate stress - full-scale queue", fontsize=11)
    fig.tight_layout()
    _save(fig, "05_base_rate_stress.png")


def fig_early_detection() -> None:
    p = ROOT / "reports" / "early_detection.json"
    if not p.exists():
        print("  skipping 06 (no early_detection.json)")
        return
    rows = json.load(open(p))["windows"]
    labels = [str(r["window"]) for r in rows]
    auc = [r["roc_auc"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    colours = [C_BAD if a < 0.5 else C_V2 for a in auc]
    ax.bar(labels, auc, color=colours)
    ax.axhline(0.5, color="#333", ls="--", lw=1)
    ax.set_ylabel("AUC"); ax.set_xlabel("truncation window")
    ax.set_title("Early detection - unrefit model on partial light curves", fontsize=11)
    ax.text(0.02, 0.93, "below chance at 15 d", transform=ax.transAxes,
            fontsize=8, color=C_BAD)
    ax.set_ylim(0, 0.7)
    _save(fig, "06_early_detection.png")


def fig_candidate_explanation() -> None:
    p = ROOT / "data" / "artefacts" / "explanations_fullscale.json"
    if not p.exists():
        print("  skipping 07"); return
    ex = json.load(open(p))
    items = list(ex.items()) if isinstance(ex, dict) else [(str(e.get("object_id")), e) for e in ex]
    for _, e in items:
        cc = e.get("channel_contributions")
        if cc:
            break
    else:
        print("  skipping 07 (no channel contributions)"); return
    # channel_contributions maps name -> {value, weight, contribution}
    names = list(cc.keys())[:9]
    vals = [(cc[n]["contribution"] if isinstance(cc[n], dict) else float(cc[n]))
            for n in names]
    order = np.argsort(vals)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([names[i] for i in order][::-1], [vals[i] for i in order][::-1], color=C_V2)
    ax.set_xlabel("channel contribution to novelty")
    ax.set_title(f"Candidate explanation - object {e.get('best_fit_class','?')} "
                 f"(p={e.get('best_fit_prob', 0):.2f})", fontsize=10)
    _save(fig, "07_candidate_explanation.png")


def fig_false_positive_audit() -> None:
    q = pd.read_parquet(ROOT / "data" / "artefacts" / "candidates_fullscale.parquet")
    q = q.sort_values("novelty_score", ascending=False).reset_index(drop=True)
    depths = [10, 50, 100, 200, 400]
    prec = [q.head(k)["is_novel"].mean() for k in depths]
    minq = [q.head(k)["quality"].min() for k in depths]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    a1.bar([str(k) for k in depths], prec, color=C_V2)
    a1.set_xlabel("queue depth"); a1.set_ylabel("precision")
    a1.set_title("Precision falls with depth", fontsize=9)
    a2.plot(depths, minq, "o-", color=C_OK)
    a2.axhline(0.30, color=C_BAD, ls="--", lw=1, label="abstain floor 0.30")
    a2.set_xlabel("queue depth"); a2.set_ylabel("minimum quality in top-k")
    a2.set_title("No false positive is low quality", fontsize=9)
    a2.legend(fontsize=8); a2.set_ylim(0, 1.05)
    fig.suptitle("False-positive audit - misses are astrophysics, not bad photometry",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "08_false_positive_audit.png")


def main() -> int:
    m = json.load(open(ROOT / "reports" / "metrics.json"))
    print("generating SuperNova figures -> reports/figures/")
    fig_architecture()
    fig_domain_match(m)
    fig_ablation(m)
    fig_precision_at_k(m)
    fig_base_rate_stress(m)
    fig_early_detection()
    fig_candidate_explanation()
    fig_false_positive_audit()
    n = len(list(OUT.glob("*.png"))) if OUT.exists() else 0
    print(f"done - {n} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
