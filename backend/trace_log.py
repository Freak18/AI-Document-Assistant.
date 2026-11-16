
import threading
import time

_lock = threading.Lock()
_logs: list[dict] = []


def log(message: str) -> dict:
    # One ordered buffer feeds both the chat response and the polling console.
    with _lock:
        entry = {
            "id": (_logs[-1]["id"] + 1) if _logs else 1,
            "line": f"{time.strftime('%H:%M:%S')} {message}",
        }
        _logs.append(entry)
        if len(_logs) > 500:
            del _logs[:100]
        return dict(entry)


def since(after_id: int) -> list[dict]:
    with _lock:
        return [dict(entry) for entry in _logs if entry["id"] > after_id]


def latest_id() -> int:
    with _lock:
        return _logs[-1]["id"] if _logs else 0
