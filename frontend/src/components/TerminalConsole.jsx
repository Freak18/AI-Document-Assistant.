import { Check, Copy, Terminal, Trash2 } from "lucide-react"
import { useEffect, useRef, useState } from "react"

function resolveStep(line) {
  const text = line.toLowerCase()
  if (text.includes("gateway")) return "Gateway"
  if (text.includes("guardrail")) return "Guardrail"
  if (text.includes("rag") || text.includes("chunk")) return "RAG"
  if (text.includes("llm") || text.includes("model") || text.includes("payload")) return "LLM"
  if (text.includes("agent") || text.includes("deep agent")) return "Deep Agent"
  if (text.includes("evaluator") || text.includes("eval")) return "Evaluation"
  if (text.includes("ingest") || text.includes("indexed")) return "Ingestion"
  return "System"
}

function resolveStatus(line) {
  const text = line.toLowerCase()
  if (text.includes("[warn]") || text.includes("blocked") || text.includes("error")) return "failed"
  if (text.includes("payload") || text.includes("retrieved document context") || text.includes("retrieved rag context") || text.includes("rag chunk")) return "payload"
  if (text.includes("completed") || text.includes("passed") || text.includes("hit") || text.includes("ready")) return "passed"
  if (text.includes("queued") || text.includes("invoking") || text.includes("activated")) return "running"
  return "info"
}

export default function TerminalConsole({ logs, onClear }) {
  const endRef = useRef(null)
  const [filter, setFilter] = useState("all")
  const [autoScroll, setAutoScroll] = useState(true)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (autoScroll) {
      endRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" })
    }
  }, [logs, autoScroll])

  const filteredLogs = logs.filter((entry) => {
    if (filter === "all") return true
    if (filter === "warn") return entry.line.includes("[WARN]") || entry.line.includes("blocked") || entry.line.includes("error")
    if (filter === "rag") return entry.line.includes("RAG") || entry.line.includes("pgvector") || entry.line.includes("Vector") || entry.line.includes("Ingested") || entry.line.includes("chunk") || entry.line.includes("Retrieved") || entry.line.includes("Payload")
    if (filter === "eval") return entry.line.includes("Evaluator") || entry.line.includes("Validation") || entry.line.includes("eval")
    if (filter === "gateway") return entry.line.includes("Gateway") || entry.line.includes("Guardrail") || entry.line.includes("cache") || entry.line.includes("agent") || entry.line.includes("LLM")
    return true
  })

  function handleCopy() {
    const text = filteredLogs.map((l) => l.line).join("\n")
    navigator.clipboard?.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  function formatLogLine(line) {
    const parts = line.split(" ")
    const time = parts[0]
    const rest = line.substring(time.length + 1)
    const step = resolveStep(rest)
    const status = resolveStatus(rest)

    let color = "text-stone-600"
    if (status === "failed") {
      color = "text-rose-600 font-medium"
    } else if (status === "passed") {
      color = "text-emerald-700 font-semibold"
    } else if (status === "running") {
      color = "text-sky-700 font-medium"
    } else if (status === "payload") {
      color = "text-slate-800"
    }

    const isPayload = status === "payload"

    return (
      <div key={line} className={`border-b border-stone-100 py-1.5 last:border-b-0 ${isPayload ? "bg-slate-50/60 px-2 py-2 my-1 rounded-xl border border-indigo-100" : ""}`}>
        <div className="mb-0.5 flex items-center gap-2">
          <span className={`rounded-full px-1.5 py-0.5 text-[9px] uppercase tracking-[0.12em] ${
            isPayload ? "border border-indigo-200 bg-indigo-50 text-indigo-700 font-semibold" : "border border-stone-200 bg-stone-100 text-stone-600"
          }`}>
            {step}
          </span>
          <span
            className={`rounded-full px-1.5 py-0.5 text-[9px] uppercase tracking-[0.12em] ${
              status === "failed"
                ? "bg-rose-100 text-rose-700"
                : status === "passed"
                ? "bg-emerald-100 text-emerald-700"
                : status === "running"
                ? "bg-sky-100 text-sky-700"
                : isPayload
                ? "bg-indigo-100/70 text-indigo-800 font-medium"
                : "bg-stone-100 text-stone-600"
            }`}
          >
            {isPayload ? "RAG Context" : status}
          </span>
          <span className="font-mono text-[10px] text-stone-400">{time}</span>
        </div>
        <div className={`whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed ${color}`}>{rest}</div>
      </div>
    )
  }

  return (
    <section className="flex max-h-[440px] min-h-[300px] flex-1 flex-col overflow-hidden rounded-[24px] border border-stone-200 bg-[#f9f9f7] shadow-[0_12px_30px_rgba(15,23,42,0.03)]">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-stone-200 px-3 py-2 text-xs">
        <div className="flex items-center gap-2 text-stone-700">
          <Terminal size={14} className="text-slate-600" />
          <span className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-600">Console</span>
          <span className="rounded-full bg-stone-200 px-2 py-0.5 text-[10px] text-stone-600">
            {filteredLogs.length} events
          </span>
        </div>

        <div className="flex items-center gap-1">
          {[
            { id: "all", label: "All" },
            { id: "gateway", label: "Flow" },
            { id: "rag", label: "Data" },
            { id: "eval", label: "Eval" },
            { id: "warn", label: "Alerts" },
          ].map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => setFilter(item.id)}
              className={`rounded-md px-1.5 py-0.5 text-[10px] uppercase transition-colors ${
                filter === item.id
                  ? "bg-white text-slate-700 shadow-sm ring-1 ring-stone-200"
                  : "text-stone-500 hover:bg-stone-100 hover:text-slate-700"
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={handleCopy}
            title="Copy logs"
            className="rounded-md p-1 text-stone-500 hover:bg-white hover:text-slate-700"
          >
            {copied ? <Check size={13} className="text-emerald-600" /> : <Copy size={13} />}
          </button>
          {onClear && (
            <button
              type="button"
              onClick={onClear}
              title="Clear console"
              className="rounded-md p-1 text-stone-500 hover:bg-white hover:text-rose-600"
            >
              <Trash2 size={13} />
            </button>
          )}
          <button
            type="button"
            onClick={() => setAutoScroll(!autoScroll)}
            title={autoScroll ? "Pause autoscroll" : "Enable autoscroll"}
            className={`rounded-md px-1.5 py-0.5 text-[10px] font-mono ${
              autoScroll ? "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200" : "text-stone-500 hover:text-slate-700"
            }`}
          >
            {autoScroll ? "On" : "Off"}
          </button>
        </div>
      </div>

      <div className="terminal-scroll flex-1 overflow-y-auto bg-white p-3 font-mono text-[11px]">
        {filteredLogs.length === 0 ? (
          <p className="italic text-stone-400">Waiting for activity…</p>
        ) : (
          filteredLogs.map((entry) => (
            <div key={entry.id}>{formatLogLine(entry.line)}</div>
          ))
        )}
        <div ref={endRef} />
      </div>
    </section>
  )
}
