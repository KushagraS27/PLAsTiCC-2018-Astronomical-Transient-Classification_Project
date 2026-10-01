import React from 'react'

function Stat({ label, value, sub }) {
  return (
    <div className="card stat">
      <span className="value">{value ?? '—'}</span>
      <span className="label">{label}</span>
      {sub && <span className="sub">{sub}</span>}
    </div>
  )
}

export default function SummaryCards({ summary }) {
  if (!summary) return null
  const tiers = summary.tiers || {}
  const bench = (summary.benchmarks || {}).nested_v2 || (summary.benchmarks || {}).baseline_v1 || {}
  const fmt = (v, d = 3) => (v == null ? '—' : Number(v).toFixed(d))

  return (
    <div className="grid cols-4">
      <Stat
        label="Candidates"
        value={summary.n_candidates?.toLocaleString()}
        sub={`${tiers.critical || 0} critical · ${tiers.high || 0} high · ${tiers.moderate || 0} moderate`}
      />
      <Stat
        label="Abstained"
        value={summary.n_abstained?.toLocaleString()}
        sub={`${((summary.abstention_rate || 0) * 100).toFixed(1)}% withheld for data quality`}
      />
      <Stat
        label="Ranking AUC"
        value={fmt(bench.roc_auc)}
        sub={bench.roc_auc_ci ? `95% CI ${fmt(bench.roc_auc_ci.low)}–${fmt(bench.roc_auc_ci.high)}` : 'no CI recorded'}
      />
      <Stat
        label="Base rate"
        value={summary.base_rate != null ? `${(summary.base_rate * 100).toFixed(2)}%` : '—'}
        sub={summary.dataset?.name || 'dataset unrecorded'}
      />
    </div>
  )
}
