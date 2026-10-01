import React, { useEffect, useState } from 'react'
import { api } from '../api.js'

function fmt(v, d = 3) {
  if (v == null || Number.isNaN(Number(v))) return '—'
  return Number(v).toFixed(d)
}

function Row({ label, m }) {
  if (!m) return null
  const ci = m.roc_auc_ci
  return (
    <tr>
      <td>{label}</td>
      <td className="num">{fmt(m.roc_auc)}{ci ? <span className="note"> [{fmt(ci.low, 2)}, {fmt(ci.high, 2)}]</span> : ''}</td>
      <td className="num">{fmt(m.average_precision)}</td>
      <td className="num">{fmt(m.p_at_10 ?? m['P@10'], 2)}</td>
      <td className="num">{fmt(m.p_at_50 ?? m['P@50'], 2)}</td>
      <td className="num">{fmt(m.lift_at_50 ?? m['lift@50'], 2)}×</td>
    </tr>
  )
}

export default function EvaluationPanel() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.evaluation().then(setData).catch((e) => setError(e.message))
  }, [])

  if (error) {
    return (
      <div className="card">
        <h2>Evaluation</h2>
        <div className="note">metrics.json not built yet — run scripts/03_evaluate.py.</div>
      </div>
    )
  }
  if (!data) return <div className="card"><h2>Evaluation</h2><div className="loading">loading…</div></div>

  const benchmarks = data.benchmarks || {}
  const stress = data.base_rate_stress?.rows || data.stress?.rows || []
  const channels = data.channel_power || {}

  return (
    <div className="card">
      <h2>Evaluation</h2>

      <div className="note" style={{ marginBottom: 10 }}>
        Locked held-out test split. Every number carries dataset{' '}
        <span className="mono">{data.dataset?.name || '?'}</span>, manifest{' '}
        <span className="mono">{data.dataset?.manifest_id || '?'}</span>, seed{' '}
        <span className="mono">{data.dataset?.seed ?? '?'}</span> and base rate{' '}
        <span className="mono">{fmt(data.base_rate, 4)}</span>.
      </div>

      <table>
        <thead>
          <tr>
            <th>ranking</th><th className="num">AUC [95% CI]</th><th className="num">AP</th>
            <th className="num">P@10</th><th className="num">P@50</th><th className="num">lift@50</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(benchmarks).map(([name, m]) => <Row key={name} label={name} m={m} />)}
        </tbody>
      </table>

      {stress.length > 0 && (
        <>
          <h3 style={{ marginTop: 18 }}>Base-rate stress</h3>
          <table>
            <thead>
              <tr>
                <th>prior</th><th className="num">AUC</th><th className="num">AP</th>
                <th className="num">P@100</th><th className="num">lift@100</th>
              </tr>
            </thead>
            <tbody>
              {stress.map((r) => (
                <tr key={r.target_rate} style={{ cursor: 'default' }}>
                  <td>{(r.target_rate * 100).toFixed(2)}%</td>
                  <td className="num">{fmt(r.roc_auc)}</td>
                  <td className="num">{fmt(r.average_precision)}</td>
                  <td className="num">{fmt(r['P@100'] ?? r.p_at_100)}</td>
                  <td className="num">{fmt(r['lift@100'] ?? r.lift_at_100, 1)}×</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="note">
            Precision falls as the prior falls. An enriched-stream precision is not a deployment number.
          </div>
        </>
      )}

      {Object.keys(channels).length > 0 && (
        <>
          <h3 style={{ marginTop: 18 }}>Channel power (locked test, not used for tuning)</h3>
          <table>
            <thead><tr><th>channel</th><th className="num">AUC</th><th className="num">AP</th><th className="num">weight</th></tr></thead>
            <tbody>
              {Object.entries(channels)
                .sort((a, b) => (b[1].roc_auc || 0) - (a[1].roc_auc || 0))
                .map(([name, m]) => (
                  <tr key={name} style={{ cursor: 'default' }}>
                    <td className="mono">{name}</td>
                    <td className="num">{fmt(m.roc_auc)}</td>
                    <td className="num">{fmt(m.average_precision)}</td>
                    <td className="num">{fmt(m.weight, 2)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  )
}
