import React, { useEffect, useState } from 'react'
import { api } from '../api.js'

export default function TaxonomyPanel() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.classes().then(setData).catch((e) => setError(e.message))
  }, [])

  if (error) return <div className="card"><h2>Taxonomy</h2><div className="note">{error}</div></div>
  if (!data) return <div className="card"><h2>Taxonomy</h2><div className="loading">loading…</div></div>

  const rows = (data.classes || []).slice().sort((a, b) => a.code - b.code)

  return (
    <div className="card">
      <h2>Taxonomy and held-out design</h2>
      <div className="note" style={{ marginBottom: 10 }}>
        {data.n_known} populations form the known-physics prior. {data.n_held_out} are withheld
        entirely — they never reach fitting, weight selection, threshold tuning or feature selection.
      </div>
      <table>
        <thead>
          <tr>
            <th className="num">code</th><th>population</th><th>family</th>
            <th className="num">train</th><th className="num">test</th><th>role</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((c) => (
            <tr key={c.code} style={{ cursor: 'default' }}>
              <td className="num">{c.code}</td>
              <td>{c.name}</td>
              <td>{c.family}</td>
              <td className="num">{c.n_train?.toLocaleString() ?? '—'}</td>
              <td className="num">{c.n_test?.toLocaleString() ?? '—'}</td>
              <td>
                {c.held_out
                  ? <span className="pill heldout">held out</span>
                  : <span className="pill routine">prior</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="note">{data.note}</div>
    </div>
  )
}
