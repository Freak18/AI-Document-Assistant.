import {
  ArrowDown,
  Bot,
  CheckCircle2,
  Database,
  Gauge,
  Shield,
  Sparkles,
} from "lucide-react"

const NODE_DEFS = {
  gateway: {
    label: "Request Intake",
    sub: "Incoming prompt",
    icon: Gauge,
  },
  guardrail: {
    label: "Guardrail Check",
    sub: "Safety gate",
    icon: Shield,
  },
  rag: {
    label: "RAG Retrieval",
    sub: "Context lookup",
    icon: Database,
  },
  llm: {
    label: "LLM Response",
    sub: "Model generation",
    icon: Sparkles,
  },
  agent: {
    label: "Deep Agent",
    sub: "Rollback / tools",
    icon: Bot,
  },
  eval: {
    label: "Validation",
    sub: "Outcome check",
    icon: CheckCircle2,
  },
}

function getNodeStyles(status) {
  if (status === "block" || status === "error") {
    return {
      border: "border-rose-200 shadow-[0_0_0_1px_rgba(254,202,202,0.7)]",
      bg: "bg-rose-50",
      text: "text-rose-700",
      pill: "bg-rose-100 text-rose-700 border-rose-200",
      badge: status === "error" ? "Failed" : "Blocked",
      glow: "ring-1 ring-rose-100",
    }
  }
  if (status === "hit") {
    return {
      border: "border-sky-200 shadow-[0_0_0_1px_rgba(186,230,253,0.7)]",
      bg: "bg-sky-50",
      text: "text-sky-700",
      pill: "bg-sky-100 text-sky-700 border-sky-200",
      badge: "Cache hit",
      glow: "ring-1 ring-sky-100",
    }
  }
  if (status === "escalated") {
    return {
      border: "border-amber-200 shadow-[0_0_0_1px_rgba(253,230,138,0.7)]",
      bg: "bg-amber-50",
      text: "text-amber-700",
      pill: "bg-amber-100 text-amber-700 border-amber-200",
      badge: "Escalated",
      glow: "ring-1 ring-amber-100",
    }
  }
  if (status === "queued") {
    return {
      border: "border-amber-200 shadow-[0_0_0_1px_rgba(253,230,138,0.7)]",
      bg: "bg-amber-50",
      text: "text-amber-700",
      pill: "bg-amber-100 text-amber-700 border-amber-200",
      badge: "Queued",
      glow: "animate-pulse ring-1 ring-amber-100",
    }
  }
  if (status === "done" || status === "pass" || status === "hit" || status === "miss" || status === "queued" || (status && status !== "idle" && status !== "skipped")) {
    return {
      border: "border-emerald-200 shadow-[0_0_0_1px_rgba(187,247,208,0.7)]",
      bg: "bg-emerald-50",
      text: "text-emerald-700",
      pill: "bg-emerald-100 text-emerald-700 border-emerald-200",
      badge: status === "active" ? "Passing" : status === "queued" ? "Queued" : status === "miss" ? "Miss" : status === "pass" ? "Passed" : status === "hit" ? "Hit" : status,
      glow: "ring-1 ring-emerald-100",
    }
  }
  if (status === "skipped") {
    return {
      border: "border-stone-200 opacity-60",
      bg: "bg-stone-50",
      text: "text-stone-500",
      pill: "bg-stone-100 text-stone-500 border-stone-200",
      badge: "Skipped",
      glow: "",
    }
  }
  return {
    border: "border-stone-200 bg-stone-50 opacity-80",
    bg: "bg-stone-50",
    text: "text-stone-500",
    pill: "bg-stone-100 text-stone-500 border-stone-200",
    badge: "Idle",
    glow: "",
  }
}

function NodeCard({ id, status, meta }) {
  const def = NODE_DEFS[id]
  const Icon = def.icon
  const style = getNodeStyles(status)
  const isLit = status && status !== "idle" && status !== "skipped"

  let detail = null
  if (meta?.detail) {
    detail = meta.detail
  } else if (meta?.stage) {
    detail = meta.stage
  } else if (id === "gateway" && meta?.cached !== undefined) {
    detail = meta.cached ? `Cached response · score ${meta.score || 1.0}` : "Cache miss · continuing"
  } else if (id === "guardrail" && meta?.reason) {
    detail = `Safety check: ${meta.reason}`
  } else if (id === "rag") {
    if (meta?.solvable) {
      detail = `Best match: ${meta.top_score} (${meta.hits} hits)`
    } else if (meta?.reason) {
      detail = `Fallback path: ${meta.reason}`
    }
  } else if (id === "llm" && meta?.model) {
    detail = `Model: ${meta.model}`
  } else if (id === "agent" && meta?.tool) {
    detail = `Tool: ${meta.tool}`
  } else if (id === "eval" && meta?.score !== undefined) {
    detail = `Quality score: ${meta.score} / 1.0`
  }

  return (
    <div
      className={`relative rounded-2xl border p-2.5 transition-all duration-300 ${style.border} ${style.bg} ${style.glow}`}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <div
            className={`flex h-7 w-7 items-center justify-center rounded-xl ${
              isLit ? "bg-white text-slate-700" : "bg-stone-100 text-stone-500"
            }`}
          >
            <Icon size={14} className={isLit ? "text-slate-700" : "text-stone-500"} />
          </div>
          <div>
            <h3 className="text-xs font-semibold leading-tight text-slate-700">{def.label}</h3>
            <p className="text-[10px] text-stone-500">{def.sub}</p>
          </div>
        </div>
        <span
          className={`rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.12em] ${style.pill}`}
        >
          {style.badge}
        </span>
      </div>
      {detail && (
        <div className="mt-1.5 border-t border-stone-200 pt-1">
          <p className="truncate font-mono text-[11px] text-stone-600">{detail}</p>
        </div>
      )}
    </div>
  )
}

function ArrowConnector({ label }) {
  return (
    <div className="my-0.5 flex items-center justify-center">
      <div className="flex items-center gap-1 text-[10px] uppercase tracking-[0.12em] text-stone-500">
        <ArrowDown size={13} className="text-stone-400" />
        {label && <span>{label}</span>}
      </div>
    </div>
  )
}

export default function ArchitectureGraph({ nodes, nodeMeta, trace, busy }) {
  return (
    <section className="flex max-h-[440px] min-h-[300px] flex-col overflow-hidden rounded-[24px] border border-stone-200 bg-white p-3.5 shadow-[0_12px_30px_rgba(15,23,42,0.03)]">
      <div className="mb-2.5 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div
            className={`flex h-2.5 w-2.5 rounded-full ${
              busy ? "bg-amber-400 animate-pulse" : "bg-emerald-400"
            }`}
          />
          <h2 className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-600">
            Flow
          </h2>
        </div>
        <span className="rounded-full border border-stone-200 bg-stone-100 px-2 py-0.5 text-[10px] text-stone-600">
          {busy ? "running" : "ready"}
        </span>
      </div>

      <div className="flex-1 overflow-y-auto pr-1">
        <div className="flex flex-col gap-1">
          <NodeCard id="gateway" status={nodes.gateway} meta={nodeMeta?.gateway} />
          <ArrowConnector label="route" />
          <NodeCard id="guardrail" status={nodes.guardrail} meta={nodeMeta?.guardrail} />
          <ArrowConnector label="check" />
          <NodeCard id="rag" status={nodes.rag} meta={nodeMeta?.rag} />

          <div className="my-1.5 grid grid-cols-2 gap-2">
            <div className="flex flex-col">
              <div className="rounded-xl border border-emerald-200 bg-emerald-50 px-2 py-1 text-[10px] uppercase tracking-[0.12em] text-emerald-700">
                answer
              </div>
            </div>
            <div className="flex flex-col">
              <div className="rounded-xl border border-amber-200 bg-amber-50 px-2 py-1 text-[10px] uppercase tracking-[0.12em] text-amber-700">
                tools
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2">
            <NodeCard id="llm" status={nodes.llm} meta={nodeMeta?.llm} />
            <NodeCard id="agent" status={nodes.agent} meta={nodeMeta?.agent} />
          </div>
          <ArrowConnector label="review" />
          <NodeCard id="eval" status={nodes.eval} meta={nodeMeta?.eval} />
        </div>

        {trace && trace.length > 0 && (
          <div className="mt-3 rounded-2xl border border-stone-200 bg-stone-50 px-2 py-2">
            <p className="mb-1 text-[10px] uppercase tracking-[0.14em] text-stone-500">Execution</p>
            <div className="flex flex-wrap gap-1.5">
              {trace.map((step) => (
                <span
                  key={step}
                  className="rounded-full border border-stone-200 bg-white px-1.5 py-0.5 text-[10px] font-medium text-stone-600"
                >
                  {step}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
