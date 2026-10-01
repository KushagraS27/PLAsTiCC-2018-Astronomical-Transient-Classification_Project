import React, { useEffect, useState } from 'react'
import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { api } from '../api.js'

// PLAsTiCC passband index -> display colour, matching the survey's ugrizy order.
const BANDS = [
  { id: 0, name: 'u', color: '#8b5cf6' },
  { id: 1, name: 'g', color: '#4ade80' },
  { id: 2, name: 'r', color: '#f87171' },
  { id: 3, name: 'i', color: '#fbbf24' },
  { id: 4, name: 'z', color: '#6ea8fe' },
  { id: 5, name: 'y', color: '#f472b6' },
]

function EvidenceBars({ evidence, weights }) {
  const entries = Object.entries(evidence || {}).sort((a, b) => b[1] - a[1])
  return (
    <div>
      {entries.map(([name, value]) => {
        const w = weights?.[name] ?? 0
        return (
          <div className="evidence-bar" key={name}>
            <span className="name" title={`production weight ${w}`}>
              {name}{w > 0 ? '' : ' · w=0'}
            </span>
            <span className="track">
              <span
                className={`fill ${w > 0 ? '' : 'zero'}`}
                style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
              />
            </span>
            <span className="val">{Number(value).toFixed(2)}</span>
          </div>
        )
      })}
    </div>
  )
}

function LightCurve({ objectId }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (objectId == null) return
    setData(null)
    api.lightcurve(objectId)
      .then(setData)
      .catch((e) => setError(e.message))
  }, [objectId])

  if (error) return <div className="error">{error}</div>
  if (!data) return <div className="loading">loading light curve…</div>

  const points = data.points || []
  const byBand = BANDS.map((b) => ({
    ...b,
    pts: points.filter((p) => p.passband === b.id && p.detected),
  })).filter((b) => b.pts.length)

  // Recharts needs one key per series on a shared x axis, so pivot into rows.
  const xs = [...new Set(points.filter((p) => p.detected).map((p) => p.mjd))].sort((a, b) => a - b)
  const rows = xs.map((x) => {
    const row = { mjd: x }
    byBand.forEach((b) => {
      const hit = b.pts.find((p) => p.mjd === x)
      if (hit) row[b.name] = hit.flux
    })
    return row
  })

  return (
    <div>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={rows} margin={{ top: 6, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid stroke="#dfe3ea" strokeDasharray="3 3" />
          <XAxis dataKey="mjd" stroke="#5b6779" tick={{ fontSize: 11, fill: "#5b6779" }} domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#5b6779" tick={{ fontSize: 11, fill: "#5b6779" }} width={52} />
          <Tooltip
            contentStyle={{ background: '#ffffff', border: '1px solid #dfe3ea', color: '#1a2230', fontSize: 12 }}
            labelFormatter={(v) => `MJD ${Number(v).toFixed(2)}`}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {byBand.map((b) => (
            <Line
              key={b.name}
              type="linear"
              dataKey={b.name}
              stroke={b.color}
              strokeWidth={1.5}
              dot={{ r: 2.5, fill: b.color, strokeWidth: 0 }}
              connectNulls={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <div className="note">
        {data.n_detections} detections of {data.n_points} epochs ·
        MJD {Number(data.mjd_range?.[0] ?? 0).toFixed(1)}–{Number(data.mjd_range?.[1] ?? 0).toFixed(1)}
      </div>
    </div>
  )
}

export default function CandidateDetail({ candidate, onVerdict }) {
  const [verdict, setVerdict] = useState('unclear')
  const [comment, setComment] = useState('')
  const [status, setStatus] = useState(null)

  if (!candidate) {
    return (
      <div className="card">
        <h2>Candidate detail</h2>
        <div className="note">Select a row from the queue to inspect its evidence channels, nearest known analogues and light curve.</div>
      </div>
    )
  }

  const c = candidate
  const explanation = c.explanation || {}
  // Field names match the actual output of explain.explain_candidate().
  const analogues = explanation.analogs || c.analogs || []
  const driver = explanation.driving_channel
  const summaryText = explanation.human_summary

  const submit = () => {
    setStatus(null)
    api.submitFeedback({ object_id: c.object_id, verdict, comment })
      .then(() => { setStatus('recorded'); setComment(''); onVerdict?.() })
      .catch((e) => setStatus(`failed: ${e.message}`))
  }

  return (
    <div className="card">
      <h2>
        Object <span className="mono">{c.object_id}</span>{' '}
        <span className={`pill ${c.tier || 'routine'}`}>{c.tier || 'routine'}</span>
        {c.abstain ? <span className="pill abstained">abstained</span> : null}
      </h2>

      <div className="banner" style={{ marginBottom: 14 }}>
        {c.disclaimer ||
          'Candidate anomaly — poorly explained by current known populations. Requires expert follow-up.'}
      </div>

      {c.abstain_reason && (
        <div className="note" style={{ marginBottom: 12, color: 'var(--bad)' }}>
          Abstained: {c.abstain_reason}
        </div>
      )}

      <dl className="kv">
        <dt>Novelty score</dt>
        <dd>{c.novelty_score?.toFixed(4)} <span className="note">({c.novelty_score_normalised?.toFixed(3)} normalised)</span></dd>
        <dt>Quality</dt>
        <dd>{c.quality?.toFixed(3)}</dd>
        <dt>Confidence</dt>
        <dd>{c.confidence?.toFixed(3)}</dd>
        <dt>Uncertainty</dt>
        <dd>{c.uncertainty?.toFixed(3)}</dd>
        <dt>Best fit</dt>
        <dd>{c.class_name} ({c.best_fit_code}) at {(c.best_fit_prob * 100)?.toFixed(1)}%</dd>
        <dt>Runner-up</dt>
        <dd>{c.runner_up_code} at {(c.runner_up_prob * 100)?.toFixed(1)}%</dd>
        <dt>Leading channel</dt>
        <dd className="mono">{driver || '—'}</dd>
      </dl>

      <h3 style={{ marginTop: 18 }}>Evidence channels</h3>
      <EvidenceBars evidence={c.evidence} weights={c.weights} />
      <div className="note">
        Channels marked <span className="mono">w=0</span> are audit channels: measured and reported, but
        excluded from the score because they were falsified on held-out novelty (AUC ≈ 0.45–0.49).
      </div>

      {summaryText && (
        <>
          <h3 style={{ marginTop: 18 }}>Why this ranked highly</h3>
          <div style={{ fontSize: 13 }}>{summaryText}</div>
        </>
      )}

      {analogues.length > 0 && (
        <>
          <h3 style={{ marginTop: 18 }}>Nearest known analogues</h3>
          <table>
            <thead>
              <tr><th>class</th><th>object</th><th className="num">distance</th></tr>
            </thead>
            <tbody>
              {analogues.slice(0, 5).map((a, i) => (
                <tr key={i} style={{ cursor: 'default' }}>
                  <td>{a.class_name || a.class_code}</td>
                  <td className="mono">{a.object_id}</td>
                  <td className="num">{a.distance?.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {explanation.analogue_summary && <div className="note">{explanation.analogue_summary}</div>}
        </>
      )}

      <h3 style={{ marginTop: 18 }}>Light curve</h3>
      <LightCurve objectId={c.object_id} />

      <h3 style={{ marginTop: 18 }}>Operator verdict</h3>
      <div className="controls">
        {['novel', 'known', 'artifact', 'unclear'].map((v) => (
          <button key={v} className={verdict === v ? 'active' : ''} onClick={() => setVerdict(v)}>{v}</button>
        ))}
      </div>
      <div className="controls">
        <input
          style={{ flex: 1 }}
          placeholder="note (optional)"
          value={comment}
          onChange={(e) => setComment(e.target.value)}
        />
        <button onClick={submit}>record</button>
      </div>
      {status && <div className="note">{status}</div>}
      <div className="note">
        Verdicts are append-only and never retrain the model automatically.
      </div>
    </div>
  )
}
