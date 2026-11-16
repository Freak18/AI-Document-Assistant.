import { FileText } from "lucide-react"
import { useEffect, useRef, useState } from "react"
import {
  clearCache,
  fetchDocuments,
  fetchHealth,
  fetchLogs,
  fetchMetrics,
  mergeLogs,
  sendChat,
  uploadPdf,
} from "./api"
import ArchitectureGraph from "./components/ArchitectureGraph"
import ChatPanel from "./components/ChatPanel"
import TerminalConsole from "./components/TerminalConsole"
import { IDLE_NODES } from "./nodes"

let messageSeq = 1

export default function App() {
  const [mode, setMode] = useState("user")
  const [adminView, setAdminView] = useState("flow")
  const [lastModel, setLastModel] = useState("")
  const [draft, setDraft] = useState("")
  const [messages, setMessages] = useState([])
  const [logs, setLogs] = useState([])
  const [nodes, setNodes] = useState(IDLE_NODES)
  const [nodeMeta, setNodeMeta] = useState({})
  const [trace, setTrace] = useState([])
  const [metrics, setMetrics] = useState({
    total_requests: 0,
    cached_requests: 0,
    cache_hit_rate: 0,
    tokens_used: 0,
    estimated_cost: 0,
    total_chunks: 0,
    latest_eval_score: null,
  })
  const [health, setHealth] = useState(null)
  const [documents, setDocuments] = useState(null)
  const [activeTitle, setActiveTitle] = useState("Your document")
  const [busy, setBusy] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadStage, setUploadStage] = useState("Preparing upload")
  const [error, setError] = useState("")
  const logCursor = useRef(0)

  useEffect(() => {
    let ignore = false
    Promise.all([fetchHealth(), fetchDocuments()])
      .then(([healthData, docsData]) => {
        if (!ignore) {
          setHealth(healthData)
          setDocuments(docsData)
          if (docsData.documents?.[0]?.title) {
            setActiveTitle(docsData.documents[0].title)
          }
          setMetrics((m) => ({ ...m, total_chunks: docsData.total_chunks }))
        }
      })
      .catch(() => {
        if (!ignore) setHealth({ llm: "offline" })
      })
    return () => {
      ignore = true
    }
  }, [])

  useEffect(() => {
    if (mode !== "admin") return undefined
    let stop = false

    const poll = async () => {
      try {
        const [logData, metricData] = await Promise.all([
          fetchLogs(logCursor.current),
          fetchMetrics(),
        ])
        if (stop) return

        if (logData.logs?.length) {
          logCursor.current = logData.logs[logData.logs.length - 1].id
          setLogs((current) => mergeLogs(current, logData.logs))

          const evalLog = logData.logs.find((e) => e.line.includes("Evaluator score"))
          if (evalLog) {
            const match = evalLog.line.match(/score:\s*([\d.]+)/)
            if (match) {
              const scoreVal = parseFloat(match[1])
              setNodes((n) => (n.eval === "queued" ? { ...n, eval: "done" } : n))
              setNodeMeta((meta) => ({
                ...meta,
                eval: { score: scoreVal },
              }))
            }
          }
        }
        setMetrics(metricData)
      } catch {
      }
    }

    poll()
    const timer = setInterval(poll, 1800)
    return () => {
      stop = true
      clearInterval(timer)
    }
  }, [mode])

  function absorbLogs(incoming) {
    if (!incoming?.length) return
    logCursor.current = Math.max(logCursor.current, incoming[incoming.length - 1].id)
    setLogs((current) => mergeLogs(current, incoming))
  }

  async function animateTrace(finalNodes, finalMeta, path) {
    const currentNodes = { ...IDLE_NODES }
    setNodes(currentNodes)
    setTrace([])

    const progressiveTrace = []
    for (const step of path) {
      progressiveTrace.push(step)
      setTrace([...progressiveTrace])
      currentNodes[step] = finalNodes[step] || "active"
      setNodes({ ...currentNodes })
      await new Promise((resolve) => setTimeout(resolve, 140))
    }
    setNodes({ ...finalNodes })
    setNodeMeta(finalMeta || {})
  }

  async function onSend(text) {
    const message = text.trim()
    if (!message || busy) return
    setDraft("")
    setError("")
    setBusy(true)
    setNodes({
      gateway: "active",
      guardrail: "queued",
      rag: "queued",
      llm: "idle",
      agent: "idle",
      eval: "idle",
    })
    setNodeMeta({
      gateway: { stage: "Request intake", detail: "Working on your prompt" },
      guardrail: { stage: "Guardrail check", detail: "Validating the request" },
      rag: { stage: "Retrieval", detail: "Searching the indexed document" },
      llm: { stage: "Model generation", detail: "Waiting for the model" },
      agent: { stage: "Deep agent", detail: "Ready if needed" },
      eval: { stage: "Validation", detail: "Checking final answer quality" },
    })

    setMessages((current) => [
      ...current,
      { id: messageSeq++, role: "user", text: message },
    ])

    try {
      const data = await sendChat(message)
      absorbLogs(data.logs)
      setMetrics(data.metrics)
      const resolvedModel =
        data?.metrics?.provider && data?.metrics?.model
          ? `${data.metrics.provider} · ${data.metrics.model}`
          : ""
      setLastModel(resolvedModel)

      if (data.trace && data.trace.length) {
        await animateTrace(data.nodes, data.node_meta, data.trace)
      } else {
        setNodes({ ...IDLE_NODES, ...data.nodes })
        setNodeMeta(data.node_meta || {})
        setTrace(data.trace || [])
      }

      const metaParts = []
      if (data.cached) metaParts.push("Semantic Cache Hit (0 Tokens)")
      else if (data.intent) metaParts.push(`Intent: ${data.intent.toUpperCase()}`)

      if (data.blocked) metaParts.push("Status: Blocked by Guardrails")
      else metaParts.push("Status: Delivered")

      setMessages((current) => [
        ...current,
        {
          id: messageSeq++,
          role: "assistant",
          text: data.answer,
          meta: metaParts.join(" · "),
          blocked: data.blocked,
          cached: data.cached,
        },
      ])
    } catch (err) {
      setError(err.message || "Failed to complete query")
    } finally {
      setBusy(false)
    }
  }

  async function simulateUploadProgress() {
    const stages = [
      { label: "Uploading PDF", progress: 18, nodes: { gateway: "active", guardrail: "pass", rag: "idle", llm: "idle", agent: "idle", eval: "idle" }, meta: { gateway: { stage: "Upload", detail: "Receiving the document" }, guardrail: { stage: "Validation", detail: "File type accepted" }, rag: { stage: "Ingestion", detail: "Waiting for extraction" }, llm: { stage: "Embedding", detail: "Queued" }, agent: { stage: "Deep agent", detail: "Standby" }, eval: { stage: "Validation", detail: "Not yet started" } } },
      { label: "Extracting text", progress: 42, nodes: { gateway: "done", guardrail: "pass", rag: "active", llm: "idle", agent: "idle", eval: "idle" }, meta: { gateway: { stage: "Upload", detail: "PDF received" }, guardrail: { stage: "Validation", detail: "Accepted" }, rag: { stage: "Ingestion", detail: "Extracting document text" }, llm: { stage: "Embedding", detail: "Queued" }, agent: { stage: "Deep agent", detail: "Standby" }, eval: { stage: "Validation", detail: "Not yet started" } } },
      { label: "Chunking content", progress: 66, nodes: { gateway: "done", guardrail: "pass", rag: "active", llm: "idle", agent: "idle", eval: "idle" }, meta: { gateway: { stage: "Upload", detail: "Document accepted" }, guardrail: { stage: "Validation", detail: "No issues found" }, rag: { stage: "Chunking", detail: "Splitting content into retrieval windows" }, llm: { stage: "Embedding", detail: "Preparing vectors" }, agent: { stage: "Deep agent", detail: "Standby" }, eval: { stage: "Validation", detail: "Waiting for index" } } },
      { label: "Embedding & indexing", progress: 88, nodes: { gateway: "done", guardrail: "pass", rag: "active", llm: "active", agent: "idle", eval: "queued" }, meta: { gateway: { stage: "Upload", detail: "Completed" }, guardrail: { stage: "Validation", detail: "Passed" }, rag: { stage: "Embedding", detail: "Creating vector representations" }, llm: { stage: "Model generation", detail: "Embedding pipeline active" }, agent: { stage: "Deep agent", detail: "Standby" }, eval: { stage: "Validation", detail: "Indexing the document" } } },
      { label: "Document ready", progress: 100, nodes: { gateway: "done", guardrail: "pass", rag: "done", llm: "done", agent: "idle", eval: "done" }, meta: { gateway: { stage: "Upload", detail: "Completed" }, guardrail: { stage: "Validation", detail: "Passed" }, rag: { stage: "Retrieval", detail: "Document indexed and searchable" }, llm: { stage: "Embedding", detail: "Completed" }, agent: { stage: "Deep agent", detail: "Standby" }, eval: { stage: "Validation", detail: "Ready for Q&A" } } },
    ]

    for (const stage of stages) {
      setUploadProgress(stage.progress)
      setUploadStage(stage.label)
      setNodes(stage.nodes)
      setNodeMeta(stage.meta)
      await new Promise((resolve) => setTimeout(resolve, 350))
    }
  }

  async function onUpload(file) {
    setUploading(true)
    setUploadProgress(0)
    setUploadStage("Uploading PDF")
    setError("")
    const progressLoop = simulateUploadProgress()

    try {
      const data = await uploadPdf(file)
      await progressLoop
      absorbLogs(data.logs)
      setMetrics(data.metrics)
      if (data.title) {
        setActiveTitle(data.title)
      }
      setLastModel("")

      const docs = await fetchDocuments()
      setDocuments(docs)

      const chunkCount = data.chunks ?? data.total_chunks ?? 0
      const readyNodeMeta = {
        gateway: { stage: "Upload", detail: "Completed" },
        guardrail: { stage: "Validation", detail: "Passed" },
        rag: {
          stage: "RAG Retrieval",
          detail: chunkCount ? `Indexed ${chunkCount} chunks` : "Indexed document",
        },
        llm: {
          stage: "Embedding",
          detail: chunkCount ? `Created ${chunkCount} vector chunks` : "Embedding complete",
        },
        agent: { stage: "Deep agent", detail: "Standby" },
        eval: { stage: "Validation", detail: "Ready for Q&A" },
      }
      setNodeMeta(readyNodeMeta)
      setNodes({
        gateway: "done",
        guardrail: "pass",
        rag: "done",
        llm: "done",
        agent: "idle",
        eval: "done",
      })
      setTrace(["upload", "ingestion", "chunking", "embedding", "retrieval", "ready"])

      setMessages((current) => [
        ...current,
        {
          id: messageSeq++,
          role: "assistant",
          text: "Document uploaded successfully. You can now ask questions after documents upload",
          meta: chunkCount ? `Document Ready · ${chunkCount} chunks indexed` : "Document Ready",
        },
      ])
    } catch (err) {
      setError(err.message || "PDF ingestion failed")
    } finally {
      setUploading(false)
      setUploadProgress(100)
      setUploadStage("Ready")
    }
  }

  function handleClearLogs() {
    setLogs([])
  }

  async function handleClearCache() {
    try {
      const res = await clearCache()
      setMetrics(res.metrics)
      setMessages((curr) => [
        ...curr,
        {
          id: messageSeq++,
          role: "assistant",
          text: `Flushed ${res.entries_removed} semantic cache entries. Subsequent queries will run through the live pipeline again.`,
          meta: "Admin Action · Cache Cleared",
        },
      ])
    } catch (err) {
      setError(err.message || "Failed to clear cache")
    }
  }

  const providerLabel =
    health?.llm === "gemini"
      ? `Gemini API · ${health.model}`
      : health?.llm === "offline"
      ? "Offline"
      : "Local model"

  return (
    <div className="flex h-screen flex-col bg-[#f5f5f3] text-slate-800">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-stone-200 bg-white/85 px-5 py-3 shadow-[0_1px_0_rgba(15,23,42,0.02)] backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-2xl bg-[linear-gradient(135deg,#dfe7ff,#f8d9d6)] shadow-[inset_0_1px_2px_rgba(255,255,255,0.8)]">
            <span className="text-sm font-semibold text-slate-700">AI</span>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-semibold tracking-[-0.02em] text-slate-800">
                AI Tool
              </h1>
            </div>
            <p className="text-[11px] text-stone-500">Ask questions, upload files, and work cleanly</p>
          </div>
        </div>

        <div className="flex items-center gap-2.5">
          {mode === "admin" && lastModel && (
            <div className="flex items-center gap-1.5 rounded-full border border-stone-200 bg-stone-100 px-3 py-1 text-xs text-slate-600">
              <div className="h-2 w-2 rounded-full bg-emerald-400" />
              <span className="font-mono text-[11px]">{lastModel}</span>
            </div>
          )}

          <div className="flex rounded-xl border border-stone-200 bg-stone-100 p-0.5 shadow-inner">
            {["user", "admin"].map((item) => (
              <button
                key={item}
                type="button"
                onClick={() => setMode(item)}
                aria-pressed={mode === item}
                className={`rounded-lg px-3 py-1.5 text-xs font-medium capitalize transition-all ${
                  mode === item
                    ? "bg-white text-slate-800 shadow-sm ring-1 ring-stone-200"
                    : "text-stone-500 hover:text-slate-700"
                }`}
              >
                {item} mode
              </button>
            ))}
          </div>
        </div>
      </header>

      {error && (
        <div className="mx-4 mt-3 flex items-center justify-between rounded-2xl border border-rose-200 bg-rose-50 px-4 py-2 text-xs text-rose-700">
          <span>{error}</span>
          <button
            type="button"
            onClick={() => setError("")}
            className="font-medium text-rose-600 transition-colors hover:text-rose-700"
          >
            Dismiss
          </button>
        </div>
      )}

      {mode === "admin" ? (
        <main className="mx-auto flex min-h-0 w-full max-w-6xl flex-1 flex-col gap-4 px-4 pb-4 pt-3">
          <aside className="flex min-h-0 flex-1 flex-col gap-3.5 overflow-hidden">
            <div className="flex items-center justify-center rounded-2xl border border-stone-200 bg-white p-1 shadow-sm">
              {[
                { id: "flow", label: "Flow" },
                { id: "console", label: "Console" },
              ].map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setAdminView(item.id)}
                  className={`flex-1 rounded-xl px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.14em] transition-all ${
                    adminView === item.id
                      ? "bg-slate-800 text-white shadow-sm"
                      : "text-stone-500 hover:bg-stone-100 hover:text-slate-700"
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </div>

            <div className="min-h-0 flex-1">
              {adminView === "flow" ? (
                <ArchitectureGraph
                  nodes={nodes}
                  nodeMeta={nodeMeta}
                  trace={trace}
                  busy={busy}
                />
              ) : (
                <TerminalConsole logs={logs} onClear={handleClearLogs} />
              )}
            </div>
          </aside>
        </main>
      ) : (
        <main className="mx-auto flex min-h-0 w-full max-w-4xl flex-1 px-4 pb-4 pt-3">
          <ChatPanel
            messages={messages}
            draft={draft}
            setDraft={setDraft}
            onSend={onSend}
            onUpload={onUpload}
            busy={busy}
            uploading={uploading}
            showMeta={false}
            adminHint={false}
            activeDocumentTitle={activeTitle}
            mode={mode}
          />
        </main>
      )}
    </div>
  )
}
