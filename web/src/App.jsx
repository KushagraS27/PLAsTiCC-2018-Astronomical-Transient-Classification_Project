import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { api } from './api.js'
import SummaryCards from './components/SummaryCards.jsx'
import CandidateTable from './components/CandidateTable.jsx'
import CandidateDetail from './components/CandidateDetail.jsx'
import EvaluationPanel from './components/EvaluationPanel.jsx'
import TaxonomyPanel from './components/TaxonomyPanel.jsx'

const DISCLAIMER =
  'Every item in this queue is a candidate anomaly, not a confirmed discovery. ' +
  'A high score means the object is poorly explained by current known populations ' +
  'and requires expert follow-up. Nothing here establishes new physics.'

export default function App() {
  const [summary, setSummary] = useState(null)
  const [version, setVersion] = useState(null)
  const [health, setHealth] = useState(null)
  const [candidates, setCandidates] = useState([])
  const [total, setTotal] = useState(0)
  const [selected, setSelected] = useState(null)
  const [error, setError] = useState(null)

  const [tier, setTier] = useState('')
  const [family, setFamily] = useState('')
  const [abstained, setAbstained] = useState('')
  const [limit, setLimit] = useState(50)
  const [offset, setOffset] = useState(0)

  useEffect(() => {
    Promise.all([api.summary(), api.version(), api.health()])
      .then(([s, v, h]) => { setSummary(s); setVersion(v); setHealth(h) })
      .catch((e) => setError(`API unreachable: ${e.message}. Start it with uvicorn api.main:app`))
  }, [])

  const loadCandidates = useCallback(() => {
    api.candidates({ limit, offset, tier, family, abstained })
      .then((r) => { setCandidates(r.candidates || []); setTotal(r.total || 0) })
      .catch((e) => setError(`could not load candidates: ${e.message}`))
  }, [limit, offset, tier, family, abstained])

  useEffect(() => { loadCandidates() }, [loadCandidates])

  const filters = useMemo(() => ({
    tier, setTier, family, setFamily, abstained, setAbstained,
  }), [tier, family, abstained])

  if (error && !summary) return <div className="app"><div className="error">{error}</div></div>

  return (
    <div className="app">
      <header className="top">
        <h1>Cosmic Novelty Engine</h1>
        <span className="codename">{summary?.codename || ''}</span>
        <span className="version">
          v{version?.version || '?'} · seed {version?.seed ?? '?'} · {health?.candidates_loaded ?? 0} candidates loaded
        </span>
      </header>

      <div className="banner">
        <strong>Read this first.</strong> {DISCLAIMER}
      </div>

      {error && <div className="error">{error}</div>}

      <SummaryCards summary={summary} />

      <div className="grid main" style={{ marginTop: 16 }}>
        <CandidateTable
          candidates={candidates}
          total={total}
          limit={limit}
          offset={offset}
          setLimit={setLimit}
          setOffset={setOffset}
          selectedId={selected?.object_id}
          onSelect={setSelected}
          filters={filters}
        />
        <CandidateDetail candidate={selected} onVerdict={loadCandidates} />
      </div>

      <div className="grid cols-2" style={{ marginTop: 16 }}>
        <EvaluationPanel />
        <TaxonomyPanel />
      </div>

      <div className="footer">
        Cosmic Novelty Engine v{version?.version} ({version?.codename}). Provenance: dataset{' '}
        <span className="mono">{summary?.dataset?.name || 'unrecorded'}</span>, manifest{' '}
        <span className="mono">{summary?.dataset?.manifest_id || 'unrecorded'}</span>, config{' '}
        <span className="mono">{version?.config_id || 'n/a'}</span>. Base rate{' '}
        <span className="mono">{summary?.base_rate != null ? summary.base_rate.toFixed(4) : 'unrecorded'}</span>.
        The phrase "new discovery confirmed" is forbidden by design in this system's output.
      </div>
    </div>
  )
}
