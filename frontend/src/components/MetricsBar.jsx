import { Activity, Coins, RotateCcw, Sparkles, Zap } from "lucide-react"

export default function MetricsBar({ metrics, onClearCache }) {
  const provider = metrics.provider ?? "—"
  const model = metrics.model ?? "—"
  const dbStatus = metrics.db_status ?? "unknown"
  const dbColor =
    dbStatus === "ok" ? "text-emerald-600" : dbStatus === "disabled" ? "text-amber-600" : "text-rose-600"

  const cards = [
    {
      id: "total",
      label: "Queries",
      value: metrics.total_requests ?? 0,
      icon: Activity,
      sub: "Handled",
    },
    {
      id: "cache",
      label: "Cache",
      value: `${metrics.cached_requests ?? 0} (${metrics.cache_hit_rate ?? 0}%)`,
      icon: Zap,
      sub: "Fast hits",
      action: onClearCache,
    },
    {
      id: "tokens",
      label: "Tokens",
      value: Number(metrics.tokens_used ?? 0).toLocaleString(),
      icon: Coins,
      sub: "Usage",
    },
    {
      id: "eval",
      label: "Quality",
      value:
        metrics.latest_eval_score !== null && metrics.latest_eval_score !== undefined
          ? `${metrics.latest_eval_score} / 1.0`
          : "Pending",
      icon: Sparkles,
      sub: "Score",
    },
  ]

  return (
    <div className="space-y-2 px-4 pb-3">
      <div className="flex flex-wrap items-center gap-2 rounded-2xl border border-stone-200 bg-white px-3 py-2 text-[11px] shadow-sm">
        <span className="text-stone-500">
          Provider: <span className="font-semibold text-slate-800 capitalize">{provider}</span>
        </span>
        <span className="text-stone-300">•</span>
        <span className="max-w-[180px] truncate font-mono text-stone-600" title={model}>{model}</span>
        <span className="text-stone-300">•</span>
        <span className="text-stone-500">
          DB: <span className={`font-semibold ${dbColor}`}>{dbStatus}</span>
        </span>
        {metrics.fallback_provider && (
          <>
            <span className="text-stone-300">•</span>
            <span className="text-stone-500">
              Fallback: <span className="font-semibold text-amber-600 capitalize">{metrics.fallback_provider}</span>
            </span>
          </>
        )}
      </div>

      <div className="grid grid-cols-2 gap-2.5 md:grid-cols-4">
        {cards.map((card) => {
          const Icon = card.icon
          return (
            <div
              key={card.id}
              className="group relative flex flex-col justify-between rounded-2xl border border-stone-200 bg-white p-3 shadow-sm transition-all hover:border-stone-300"
            >
              <div className="flex items-center justify-between text-stone-500">
                <span className="text-[10px] font-medium uppercase tracking-[0.14em] text-stone-500">
                  {card.label}
                </span>
                <div className="flex items-center gap-1">
                  {card.action && (
                    <button
                      type="button"
                      onClick={card.action}
                      title="Clear cache"
                      className="rounded-md p-0.5 text-stone-400 transition-all hover:bg-stone-100 hover:text-slate-700"
                    >
                      <RotateCcw size={11} />
                    </button>
                  )}
                  <Icon size={14} className="text-slate-600" />
                </div>
              </div>
              <p className="mt-1.5 text-base font-semibold text-slate-800">{card.value}</p>
              <p className="mt-0.5 text-[10px] text-stone-500">{card.sub}</p>
            </div>
          )
        })}
      </div>
    </div>
  )
}
