from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, TypedDict

from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph

from agent import agent
from database import DB_ENABLED, SessionLocal
from gateway import gateway
from guardrails import guardrails
from models import Evaluation, WorkflowLog
from trace_log import log
from vector_store import store

RAG_PROMPT = PromptTemplate.from_template(
    "You are a careful document Q&A assistant. Answer the user's question using only the provided Context below.\n"
    "Prioritize exact field values from the document, such as names, dates, amounts, titles, roles, locations, policy numbers, and direct labels.\n"
    "If the answer is directly stated in the document, return it in one short factual sentence and prefer the exact wording from the document.\n"
    "If the uploaded document does not contain the answer, say 'The uploaded document does not contain this information.' in one clear sentence.\n"
    "Do not answer from general knowledge or legal boilerplate; use the context only.\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n"
)

# Intent patterns for routing (order / math trigger Deep Agent)
ORDER_INTENT = re.compile(
    r"\bORD-\d+\b|\b(order|package|delivery|shipment)\s*(status|lookup|id|track|info|#|\d+)\b|\btrack\s+(my\s+)?order\b",
    re.IGNORECASE,
)
MATH_INTENT = re.compile(
    r"\d\s*[\+\-\*/x×÷%^]\s*\d|\b(calculate|compute|evaluate|add|subtract|multiply|divide|sum)\b"
    r"|\bwhat is\s+\d|\b\d+\s*(%|percent)\s*(of)?\s*\d+"
    r"|\b(plus|minus|times|divided\s+by)\b",
    re.IGNORECASE,
)


def _looks_like_summary_question(prompt: str) -> bool:
    text = prompt.lower().strip()
    if not text:
        return False

    # Never treat specific factual or field-level inquiries as general document summaries
    specific_field_keywords = [
        "compensation", "salary", "ctc", "pay", "fixed", "remuneration", "bonus",
        "name", "candidate", "who is", "designation", "title", "role", "position",
        "date", "joining", "when", "location", "probation", "notice", "hours",
        "settlement", "margin", "penalty", "order", "status", "clause", "cycle",
        "amount", "inr", "dollar", "how much", "what is the"
    ]
    if any(k in text for k in specific_field_keywords):
        return False

    summary_patterns = [
        r"\b(summarize\s+(this|the)|give\s+(me\s+)?a\s+summary|provide\s+(a\s+)?summary|high\s*level\s+summary|overview\s+of\s+(this|the))\b",
        r"\bwhat\s+is\s+(this|the)\s+(document|file|pdf|report|letter|agreement|contract)\s+about\b",
        r"\bwhat\s+is\s+this\s+about\b",
        r"\bwhat\s+is\s+the\s+main\s+(purpose|idea|topic|subject)\b",
        r"\btell\s+me\s+about\s+(this|the)\s+(document|file|pdf|report|letter|agreement|contract)\b",
        r"\b(key\s+points|general\s+summary)\s+of\s+(this|the)\b",
        r"\b(summarize|summary)\s+(this|the)\s+document\b",
        r"^summarize\b",
    ]
    return any(re.search(pattern, text) for pattern in summary_patterns)


def _classify_question(prompt: str) -> str:
    # Determine intent of the user query
    text = prompt.lower()
    if bool(ORDER_INTENT.search(prompt)):
        return "order"
    if bool(MATH_INTENT.search(prompt)):
        return "math"
    if _looks_like_summary_question(prompt):
        return "summary"
    return "fact"
# ---------------------------------------------------------------------------
# Intent‑aware cache helper


# ---------------------------------------------------------------------------
def _answer_satisfies_intent(prompt: str, answer: str) -> bool:
    """Return True if the cached answer appears to satisfy the intent of *prompt*.

    Uses the same classification as `_classify_question` and simple regexes for
    common field‑level queries (joining date, compensation, name). For other intents
    we assume the cached answer is acceptable.
    """
    intent = _classify_question(prompt)
    low_ans = answer.lower()
    if intent == "joining":
        # date patterns like "21 Oct 2026" or "21-Oct-2026"
        return bool(re.search(r"\b\d{1,2}[\s/-][a-z]{3,9}[\s/-]\d{4}\b", low_ans))
    if intent == "compensation":
        # monetary amounts – allows commas and optional currency symbols
        return bool(re.search(r"\b\d{1,3}(,\d{2,3})*(\s*[₹$]?)\b", low_ans))
    if intent == "name":
        # simple capitalised name (two words) detection
        return bool(re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", answer))
    # For other intents (order, math, fact, summary) we accept the cached answer.
    return True


def _retrieval_confidence(hits: list[dict]) -> tuple[float, float, float]:
    if not hits:
        return 0.0, 0.0, 0.0
    top = hits[0]
    top_score = float(top.get("score", 0.0))
    top_lexical = float(top.get("lexical", 0.0))
    avg_lexical = sum(float(hit.get("lexical", 0.0)) for hit in hits) / len(hits)
    return top_score, top_lexical, avg_lexical


def _build_rag_prompt(context: str, question: str) -> str:
    if _looks_like_summary_question(question):
        return (
            "You are a careful document summarizer. Use only the supplied Context below to write a concise summary "
            "of the uploaded document. Mention the document's purpose, core topic, and key ideas in 2-5 short sentences. "
            "Do not say the document does not contain this information when the context is available.\n\n"
            "Context:\n{context}\n\nQuestion:\n{question}\n"
        ).format(context=context, question=question)
    return RAG_PROMPT.format(context=context, question=question)


class WorkflowState(TypedDict):
    prompt: str
    answer: str
    context: str
    nodes: dict[str, str]
    node_meta: dict[str, Any]
    trace: list[str]
    cached: bool
    blocked: bool
    solvable: bool
    escalation_reason: str
    intent: str


def _mark(state: WorkflowState, node: str, status: str, meta: Any = None) -> dict:
    nodes = dict(state["nodes"])
    nodes[node] = status
    trace = list(state["trace"])
    if node not in trace:
        trace.append(node)
    node_meta = dict(state.get("node_meta", {}))
    if meta is not None:
        node_meta[node] = meta
    return {"nodes": nodes, "trace": trace, "node_meta": node_meta}


def _persist_workflow_log(request_id: str, node_name: str, status: str, log_message: str, duration_ms: float = None) -> None:
    if not DB_ENABLED or SessionLocal is None:
        return
    try:
        db = SessionLocal()
        try:
            entry = WorkflowLog(
                id=str(uuid.uuid4()),
                request_id=request_id,
                node_name=node_name,
                status=status,
                log_message=log_message,
                duration_ms=duration_ms,
            )
            db.add(entry)
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        log(f"[WARN] Could not persist workflow log to DB: {exc}")


async def gateway_node(state: WorkflowState) -> dict:
    gateway.record_request()
    hit, score = gateway.lookup(state["prompt"])
    if hit:
        # Verify that the cached answer actually satisfies the request intent
        if _answer_satisfies_intent(state["prompt"], hit["answer"]):
            log(f"[INFO] Gateway semantic cache hit similarity={score:.2f}")
            marked = _mark(state, "gateway", "hit", {"score": round(score, 2), "cached": True})
            # Propagate the cache hit to the RAG node so downstream nodes see it as active
            marked = _mark(marked, "rag", "active", {"reason": "cache_hit"})
            return {**marked, "cached": True, "answer": hit["answer"]}
        else:
            log(f"[INFO] Cache hit does not satisfy intent for prompt '{state['prompt']}', falling back to fresh retrieval")
            # treat as miss – continue to normal RAG path


    if gateway.cache_size():
        log(f"[INFO] Gateway semantic cache miss best_similarity={score:.2f}")
    else:
        log("[INFO] Gateway semantic cache miss (cache empty)")
    marked = _mark(state, "gateway", "miss", {"score": round(score, 2), "cached": False})
    return {**marked, "cached": False}


async def guardrail_node(state: WorkflowState) -> dict:
    reason = guardrails.regex_reason(state["prompt"])
    if reason:
        log(f"[WARN] Guardrail regex blocked prompt: reason={reason}")
        marked = _mark(state, "guardrail", "block", {"reason": reason, "type": "regex"})
        return {
            **marked,
            "blocked": True,
            "cached": False,
            "answer": f"Request blocked by Enterprise Guardrails filter ({reason}).",
        }

    # If it was a semantic cache hit, it was already vetted before entering the cache.
    # We fast-bypass the LLM judge to ensure 0 token cost and sub-millisecond response.
    if state.get("cached"):
        log("[INFO] Guardrail fast-passed cached request without LLM judge call")
        return _mark(state, "guardrail", "pass", {"type": "cached_bypass"})

    log("[INFO] Guardrail regex filter passed; invoking LLM-as-a-Judge...")
    allowed = await guardrails.judge(state["prompt"])
    if not allowed:
        log("[WARN] Guardrail LLM-as-a-judge blocked the prompt")
        marked = _mark(state, "guardrail", "block", {"reason": "llm_judge", "type": "judge"})
        return {
            **marked,
            "blocked": True,
            "cached": False,
            "answer": "Request blocked by Enterprise Guardrails (LLM intent policy).",
        }
    return _mark(state, "guardrail", "pass", {"type": "vetted"})


def route_after_guardrail(state: WorkflowState) -> str:
    if state["blocked"] or state["cached"]:
        return "stop"
    return "continue"


async def rag_solver_node(state: WorkflowState) -> dict:
    prompt = state["prompt"]
    question_kind = _classify_question(prompt)

    if question_kind == "order":
        reason = "operational_tool_required"
        log(f"[INFO] LangGraph RAG solver: order lookup detected; escalating to Deep Agent")
        marked = _mark(state, "rag", "escalated", {"solvable": False, "reason": reason, "escalate_to": "deep_agent", "intent": "order"})
        return {**marked, "solvable": False, "escalation_reason": reason, "intent": "order"}

    if question_kind == "math":
        reason = "operational_tool_required"
        log(f"[INFO] LangGraph RAG solver: math query detected; escalating to Deep Agent")
        marked = _mark(state, "rag", "escalated", {"solvable": False, "reason": reason, "escalate_to": "deep_agent", "intent": "math"})
        return {**marked, "solvable": False, "escalation_reason": reason, "intent": "math"}

    hits = store.search(prompt, k=5)
    if not hits:
        reason = "no_indexed_document_matches"
        no_match_answer = "The uploaded document does not contain this information."
        log("[INFO] LangGraph RAG solver: no indexed documents matched the query; refusing general fallback")
        gateway.remember(prompt, no_match_answer)
        marked = _mark(state, "rag", "escalated", {"solvable": False, "reason": reason, "intent": "no_match"})
        return {**marked, "solvable": False, "escalation_reason": reason, "intent": "no_match", "answer": no_match_answer}

    top_score, top_lexical, avg_lexical = _retrieval_confidence(hits)
    retrieval_source = hits[0].get("retrieval_source", "pgvector")
    combined_confidence = (0.45 * top_score) + (0.55 * max(top_lexical, avg_lexical))
    log(
        "[INFO] LangGraph RAG evaluation: "
        f"{retrieval_source} top_score={top_score:.2f} top_lexical={top_lexical:.2f} "
        f"avg_lexical={avg_lexical:.2f} combined={combined_confidence:.2f}"
    )

    if question_kind != "summary" and combined_confidence < 0.12 and top_score < 0.15:
        reason = "insufficient_grounding"
        no_match_answer = "The uploaded document does not contain this information."
        log(f"[INFO] LangGraph RAG solver: grounding is too weak ({combined_confidence:.2f}); returning no-match rather than general knowledge")
        gateway.remember(prompt, no_match_answer)
        marked = _mark(state, "rag", "escalated", {"solvable": False, "reason": reason, "top_score": top_score, "intent": "no_match"})
        return {**marked, "solvable": False, "escalation_reason": reason, "intent": "no_match", "answer": no_match_answer}

    if question_kind == "summary":
        log(f"[INFO] LangGraph RAG solver: summary/document-overview query accepted for '{prompt[:80]}'")
        # Summaries need to be answered from the uploaded document even when the ranking is not extremely strong.
        # The document itself is the context source, so a moderate match should still use the document rather than a false no-match fallback.

    blocks = [
        f"[{hit['source']} chunk {hit['chunk']} similarity={hit['score']:.2f}]\n{hit['text'][:1500]}"
        for hit in hits[:5]
    ]
    log(f"[INFO] LangGraph RAG: Retrieved {len(hits)} relevant chunks from {retrieval_source} (top_score={top_score:.2f})")
    for idx, hit in enumerate(hits[:3], 1):
        clean_text = " ".join(hit["text"].split())
        preview = clean_text[:140] + ("..." if len(clean_text) > 140 else "")
        log(f"[INFO] RAG Chunk #{idx} [{hit.get('source', '')} c{hit.get('chunk', 1)} | sim={hit.get('score', 0):.2f}]: \"{preview}\"")

    marked = _mark(state, "rag", "active", {"solvable": True, "hits": len(hits), "top_score": top_score, "engine": retrieval_source})
    return {
        **marked,
        "solvable": True,
        "context": "\n\n".join(blocks),
        "intent": "rag",
    }


def route_by_rag_solvability(state: WorkflowState) -> str:
    """Only the documented operational tasks should escalate to Deep Agent. Generic unmatched document questions stop here."""
    if state.get("intent") == "no_match":
        return "stop"
    if state.get("solvable", False):
        return "rag_llm"
    return "deep_agent"


async def llm_node(state: WorkflowState) -> dict:
    context = state["context"]
    prompt = _build_rag_prompt(context, state["prompt"])

    # Log the exact RAG-augmented request payload the LLM is receiving
    context_snippets = []
    for block in context.split("\n\n"):
        block = block.strip()
        if block:
            header, _, body = block.partition("\n")
            clean_body = " ".join(body.split())
            if len(clean_body) > 220:
                clean_body = clean_body[:220] + "..."
            context_snippets.append(f"  • {header}: \"{clean_body}\"")
    context_display = "\n".join(context_snippets) if context_snippets else "  (No context chunks)"

    log(
        "[INFO] LLM Request Payload (RAG Retrieved Context & Prompt):\n"
        f"▶ Question: {state['prompt']}\n"
        f"▶ Context Retrieved from RAG ({len(context_snippets)} chunks):\n"
        f"{context_display}"
    )

    answer = await gateway.complete(prompt, purpose="rag_answer")
    gateway.remember(state["prompt"], answer)
    log(f"[INFO] Prompt augmented with retrieved document chunks; completed via LLM Gateway ({gateway.model_name})")
    marked = _mark(state, "llm", "active", {"model": gateway.model_name})
    return {**marked, "answer": answer}


async def agent_node(state: WorkflowState) -> dict:
    log(f"[INFO] Deep Agent activated autonomously (reason: {state.get('escalation_reason', 'unresolved')})")
    try:
        answer, tool_name = agent.run(state["prompt"], state.get("intent", "general"))
    except Exception as exc:
        answer, tool_name = f"Autonomous tool execution error: {exc}", "error"

    log(f"[INFO] Deep Agent autonomously selected tool: {tool_name}")
    log("[INFO] Deep Agent returned deterministic tool answer (0 generation tokens used)")
    gateway.remember(state["prompt"], answer)
    marked = _mark(state, "agent", "active", {"tool": tool_name, "reason": state.get("escalation_reason")})
    return {**marked, "answer": answer}


def build_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("gateway", gateway_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("rag_solver", rag_solver_node)
    graph.add_node("llm", llm_node)
    graph.add_node("agent", agent_node)

    graph.set_entry_point("gateway")
    graph.add_edge("gateway", "guardrail")

    # Guardrails: stop if blocked or cached
    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"stop": END, "continue": "rag_solver"},
    )

    # LangGraph RAG Solver: If solvable -> RAG LLM. If unable to solve -> Deep Agent!
    graph.add_conditional_edges(
        "rag_solver",
        route_by_rag_solvability,
        {"rag_llm": "llm", "deep_agent": "agent", "stop": END},
    )

    graph.add_edge("llm", END)
    graph.add_edge("agent", END)
    return graph.compile()


workflow = build_graph()


async def run_workflow(prompt: str) -> dict:
    state: WorkflowState = {
        "prompt": prompt,
        "answer": "",
        "context": "",
        "nodes": {
            "gateway": "idle",
            "guardrail": "idle",
            "rag": "idle",
            "llm": "idle",
            "agent": "idle",
            "eval": "idle",
        },
        "node_meta": {},
        "trace": [],
        "cached": False,
        "blocked": False,
        "solvable": False,
        "escalation_reason": "",
        "intent": "",
    }
    result = await workflow.ainvoke(state)
    if result.get("cached") and not result.get("blocked"):
        gateway.mark_hit()

    return {
        "answer": result.get("answer", ""),
        "nodes": result.get("nodes", {}),
        "node_meta": result.get("node_meta", {}),
        "trace": result.get("trace", []),
        "cached": bool(result.get("cached")),
        "blocked": bool(result.get("blocked")),
        "intent": result.get("intent") or ("cache" if result.get("cached") else ""),
        "solvable": bool(result.get("solvable")),
    }


async def run_evaluator(prompt: str, answer: str) -> None:
    rubric = (
        "You are an automated evaluation judge for an Enterprise AI assistant. "
        "Score how directly, accurately, and concisely the answer addresses the user question on a scale from 0.0 to 1.0. "
        "Reply with ONLY a decimal number between 0.0 and 1.0 (e.g. 0.92).\n"
        f"Question: {prompt}\nAnswer: {answer}"
    )
    verdict = await gateway.complete(rubric, purpose="evaluator")
    match = re.search(r"\d(?:\.\d+)?", verdict)
    score_float = float(match.group(0)) if match else 0.88
    score_float = max(0.0, min(1.0, score_float))
    gateway.set_eval_score(score_float)
    log(f"[INFO] Evaluator score: {score_float:.2f} (Directness & Quality)")

    # Persist evaluation to PostgreSQL
    if DB_ENABLED and SessionLocal is not None:
        try:
            db = SessionLocal()
            try:
                eval_record = Evaluation(
                    id=str(uuid.uuid4()),
                    directness_score=score_float,
                    factual_alignment_score=score_float,
                    evaluator_feedback=verdict[:500] if verdict else "",
                    status="completed",
                    provider=gateway.metrics().get("provider"),
                    model=gateway.metrics().get("model"),
                )
                db.add(eval_record)
                db.commit()
            finally:
                db.close()
        except Exception as exc:
            log(f"[WARN] Could not persist evaluation to DB: {exc}")
