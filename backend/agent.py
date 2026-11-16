from __future__ import annotations

import ast
import operator
import re

from langchain_core.tools import tool

# Fallback order data
_FALLBACK_ORDERS = {
    "ORD-1001": {"status": "Shipped", "item": "Equity delivery basket", "eta": "12 Nov 2026"},
    "ORD-1002": {"status": "Pending KYC", "item": "Demat account opening", "eta": "on hold"},
    "ORD-2044": {"status": "Settled", "item": "Listed corporate bond", "eta": "settled T+1"},
    "ORD-3180": {"status": "Rejected", "item": "Margin funding request", "eta": "not applicable"},
}

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}



def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("division by zero")
        if isinstance(node.op, ast.Pow) and abs(right) > 8:
            raise ValueError("exponent too large")
        return _BINOPS[type(node.op)](left, right)
    raise ValueError("only numbers and + - * / % ** are allowed")


def safe_calculate(expression: str) -> str:
    """AST-walk the expression so no names, calls, or arbitrary code execute."""
    tree = ast.parse(expression, mode="eval")
    value = _eval_node(tree)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(round(value, 6)).rstrip("0").rstrip(".")


def extract_expression(prompt: str) -> str:
    text = prompt.lower()

    percent_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:of)?\s*(\d+(?:\.\d+)?)",
        text,
    )
    if percent_match:
        return f"({percent_match.group(1)}/100)*{percent_match.group(2)}"

    replacements = [
        (r"\bplus\b", "+"),
        (r"\bminus\b", "-"),
        (r"\btimes\b", "*"),
        (r"\bmultiplied\s+by\b", "*"),
        (r"\bdivided\s+by\b", "/"),
        (r"\bover\b", "/"),
        (r"\bx\b", "*"),
        (r"\b×\b", "*"),
        (r"\b÷\b", "/"),
        (r"\^", "**"),
    ]
    normalized = text
    for pattern, rep in replacements:
        normalized = re.sub(pattern, rep, normalized)

    match = re.search(
        r"[-+]?[0-9]+(?:\.[0-9]+)?(?:\s*[-+*/%^]\s*[-+]?[0-9]+(?:\.[0-9]+)?)+",
        normalized,
    )
    if match:
        clean = re.sub(r"\s+", "", match.group(0))
        return clean

    match_parens = re.search(r"[0-9][0-9.\s+\-*/%()]+[0-9]", normalized)
    if match_parens:
        return re.sub(r"\s+", "", match_parens.group(0))

    match_single = re.search(r"\b\d+(?:\.\d+)?\b", normalized)
    if match_single:
        return match_single.group(0)

    raise ValueError(f"Could not parse arithmetic expression from '{prompt}'")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely."""
    return safe_calculate(expression)


# ---------------------------------------------------------------------------
def _lookup_order_db(order_number: str) -> dict | None:
    """Query the orders table in PostgreSQL."""
    from database import DB_ENABLED, SessionLocal
    if not DB_ENABLED or SessionLocal is None:
        return None
    try:
        from models import Order
        db = SessionLocal()
        try:
            row = db.query(Order).filter_by(order_number=order_number).first()
            if row:
                return {"status": row.status, "item": row.item, "eta": row.eta}
        finally:
            db.close()
    except Exception:
        pass
    return None


@tool
def order_lookup(order_id: str) -> str:
    """Look up a brokerage order by ID (e.g. ORD-1001)."""
    clean_id = order_id.strip().upper()
    if not clean_id.startswith("ORD-") and clean_id.isdigit():
        clean_id = f"ORD-{clean_id}"

    # 1. Try PostgreSQL
    row = _lookup_order_db(clean_id)

    # 2. Fall back to in-memory dict
    if row is None:
        row = _FALLBACK_ORDERS.get(clean_id)

    if row is None:
        known = ", ".join(_FALLBACK_ORDERS.keys())
        return f"Order {clean_id} was not found. Known sample IDs: {known}."
    return (
        f"Order {clean_id} is {row['status']} "
        f"for {row['item']} (ETA: {row['eta']})."
    )


# ---------------------------------------------------------------------------
# Deep Agent
# ---------------------------------------------------------------------------

    def run(self, prompt: str, intent: str) -> tuple[str, str]:
        # Math / calculation
        if intent == "math" or re.search(
            r"\d\s*[\+\-\*/x×÷%^]\s*\d|"
            r"\b(calculate|compute|add|subtract|multiply|divide|sum|percent)\b",
            prompt,
            re.IGNORECASE,
        ):
            try:
                expression = extract_expression(prompt)
                answer = calculator.invoke({"expression": expression})
                return str(answer), f"calculator({expression})"
            except Exception as exc:
                if intent == "math":
                    return f"Calculation error: {exc}", "calculator(error)"

        # Order lookup
        match = re.search(r"(?:ORD-)?(\d{4})", prompt, re.IGNORECASE)
        if intent == "order" or match or "order" in prompt.lower():
            if not match:
                known = ", ".join(_FALLBACK_ORDERS.keys())
                return f"Please provide a valid order ID such as {known}.", "order_lookup(missing id)"
            raw_num = match.group(1)
            order_id_str = f"ORD-{raw_num}"
            answer = order_lookup.invoke({"order_id": order_id_str})
            return str(answer), f"order_lookup({order_id_str})"

        # No matching tool
        known_orders = ", ".join(_FALLBACK_ORDERS.keys())
        return (
            "The uploaded document does not contain enough information for this request, "
            "and it did not match any available autonomous agent tool.\n\n"
            "• Ask about facts directly present in the document\n"
            "• Check names, dates, values, roles, or titles from the uploaded file\n"
            f"• Track operational orders if relevant (e.g., {known_orders})",
            "agent(no_matching_tool)",
        )


class DeepAgent:
    """Analyze query intent, select the best tool, and return the deterministic result."""

    def run(self, prompt: str, intent: str) -> tuple[str, str]:
        # Math / calculation
        if intent == "math" or re.search(
            r"\d\s*[\+\-\*/x×÷%^]\s*\d|"
            r"\b(calculate|compute|add|subtract|multiply|divide|sum|percent)\b",
            prompt,
            re.IGNORECASE,
        ):
            try:
                expression = extract_expression(prompt)
                answer = calculator.invoke({"expression": expression})
                return str(answer), f"calculator({expression})"
            except Exception as exc:
                if intent == "math":
                    return f"Calculation error: {exc}", "calculator(error)"

        # Order lookup
        match = re.search(r"(?:ORD-)?(\d{4})", prompt, re.IGNORECASE)
        if intent == "order" or match or "order" in prompt.lower():
            if not match:
                known = ", ".join(_FALLBACK_ORDERS.keys())
                return f"Please provide a valid order ID such as {known}.", "order_lookup(missing id)"
            raw_num = match.group(1)
            order_id_str = f"ORD-{raw_num}"
            answer = order_lookup.invoke({"order_id": order_id_str})
            return str(answer), f"order_lookup({order_id_str})"

        # No matching tool
        known_orders = ", ".join(_FALLBACK_ORDERS.keys())
        return (
            "The uploaded document does not contain enough information for this request, "
            "and it did not match any available autonomous agent tool.\n\n"
            "• Ask about facts directly present in the document\n"
            "• Check names, dates, values, roles, or titles from the uploaded file\n"
            f"• Track operational orders if relevant (e.g., {known_orders})",
            "agent(no_matching_tool)",
        )


agent = DeepAgent()
