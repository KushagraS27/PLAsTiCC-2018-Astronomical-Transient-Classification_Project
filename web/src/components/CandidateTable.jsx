import React from 'react'

const TIERS = ['', 'routine', 'moderate', 'high', 'critical']
const FAMILIES = [
  ['', 'all'],
  ['galactic', 'galactic'],
  ['extragalactic', 'extragalactic'],
]

export default function CandidateTable({
  candidates, total, limit, offset, setLimit, setOffset, selectedId, onSelect, filters,
}) {
  const { tier, setTier, family, setFamily, abstained, setAbstained } = filters
  const page = Math.floor(offset / limit) + 1
  const pages = Math.max(1, Math.ceil(total / limit))

  return (
    <div className="card">
      <h2>Candidate queue</h2>
      <div className="controls">
        <select value={tier} onChange={(e) => { setTier(e.target.value); setOffset(0) }}>
          {TIERS.map((t) => <option key={t || 'all'} value={t}>{t || 'all tiers'}</option>)}
        </select>
        <select value={family} onChange={(e) => { setFamily(e.target.value); setOffset(0) }}>
          {FAMILIES.map(([v, label]) => <option key={v || 'all'} value={v}>{label === 'all' ? 'all families' : label}</option>)}
        </select>
        <select value={abstained} onChange={(e) => { setAbstained(e.target.value); setOffset(0) }}>
          <option value="">abstained + scored</option>
          <option value="false">scored only</option>
          <option value="true">abstained only</option>
        </select>
        <select value={limit} onChange={(e) => { setLimit(Number(e.target.value)); setOffset(0) }}>
          {[25, 50, 100, 200].map((n) => <option key={n} value={n}>{n} / page</option>)}
        </select>
        <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>← prev</button>
        <span className="note" style={{ alignSelf: 'center' }}>page {page} / {pages} · {total.toLocaleString()} total</span>
        <button disabled={offset + limit >= total} onClick={() => setOffset(offset + limit)}>next →</button>
      </div>

      <table>
        <thead>
          <tr>
            <th className="num">#</th>
            <th>object</th>
            <th>tier</th>
            <th>best fit</th>
            <th className="num">novelty</th>
            <th className="num">quality</th>
            <th className="num">conf.</th>
          </tr>
        </thead>
        <tbody>
          {candidates.map((c) => (
            <tr
              key={c.object_id}
              className={selectedId === c.object_id ? 'selected' : ''}
              onClick={() => onSelect(c)}
            >
              <td className="num">{c.rank ?? '—'}</td>
              <td className="mono">{c.object_id}</td>
              <td>
                <span className={`pill ${c.tier || 'routine'}`}>{c.tier || 'routine'}</span>
                {c.abstain ? <span className="pill abstained" style={{ marginLeft: 4 }}>abstained</span> : null}
              </td>
              <td>
                {c.class_name || '—'}
                <span className="note"> {c.best_fit_prob != null ? `${(c.best_fit_prob * 100).toFixed(0)}%` : ''}</span>
              </td>
              <td className="num">{c.novelty_score_normalised != null ? c.novelty_score_normalised.toFixed(3) : '—'}</td>
              <td className="num">{c.quality != null ? c.quality.toFixed(2) : '—'}</td>
              <td className="num">{c.confidence != null ? c.confidence.toFixed(2) : '—'}</td>
            </tr>
          ))}
          {!candidates.length && (
            <tr><td colSpan={7} className="note">No candidates match these filters.</td></tr>
          )}
        </tbody>
      </table>
    </div>
  )
}
