// Every call goes to a relative /api path. Absolute URLs would break both behind
// the FastAPI static mount and behind the sandbox preview host.
async function get(path) {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`${path} -> ${res.status}`)
  return res.json()
}

export const api = {
  health: () => get('/api/health'),
  version: () => get('/api/version'),
  summary: () => get('/api/summary'),
  candidates: (params = {}) => {
    const q = new URLSearchParams(
      Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== ''),
    )
    return get(`/api/candidates?${q}`)
  },
  candidate: (id) => get(`/api/candidate/${id}`),
  lightcurve: (id) => get(`/api/lightcurve/${id}`),
  evaluation: () => get('/api/evaluation'),
  classes: () => get('/api/classes'),
  feedback: () => get('/api/feedback'),
  submitFeedback: (body) =>
    fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then((r) => {
      if (!r.ok) return r.json().then((e) => Promise.reject(new Error(e.detail || r.status)))
      return r.json()
    }),
}
