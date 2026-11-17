async function readJson(response) {
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    const detail = body.detail
    const message = typeof detail === "string" ? detail : "Request failed"
    throw new Error(message)
  }
  return body
}

export function sendChat(message, conversationId = null) {
  const payload = conversationId ? { message, conversation_id: conversationId } : { message }
  return fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }).then(readJson)
}

export function uploadPdf(file) {
  const body = new FormData()
  body.append("file", file)
  return fetch("/api/upload", { method: "POST", body }).then(readJson)
}

export function fetchLogs(after = 0) {
  return fetch(`/api/logs?after=${after}`).then(readJson)
}

export function fetchMetrics() {
  return fetch("/api/metrics").then(readJson)
}

export function fetchHealth() {
  return fetch("/api/health").then(readJson)
}

export function fetchDocuments() {
  return fetch("/api/documents").then(readJson)
}

export function clearCache() {
  return fetch("/api/clear-cache", { method: "POST" }).then(readJson)
}

export function mergeLogs(current, incoming) {
  const seen = new Set(current.map((entry) => entry.id))
  const next = [...current]
  for (const entry of incoming) {
    if (!seen.has(entry.id)) next.push(entry)
  }
  return next
}
