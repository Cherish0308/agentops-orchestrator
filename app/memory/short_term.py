"""
Short-Term Working Memory — Redis-backed, scoped per task.

During a task run every agent can read/write to this shared store.
Keys are namespaced as  task:{task_id}:{field}
The entire namespace is wiped when the task completes.

If Redis is unavailable the store silently falls back to an
in-process dict so the rest of the system keeps working.
"""
import json
import os
from typing import Any, Dict, Optional

TTL_SECONDS = 3600  # 1 hour — tasks that hang don't leak memory forever

try:
    import redis

    _client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0,
        decode_responses=True,
        socket_connect_timeout=2,
    )
    _client.ping()
    _REDIS_AVAILABLE = True
except Exception:
    _REDIS_AVAILABLE = False
    _fallback: Dict[str, str] = {}


def _key(task_id: str, field: str) -> str:
    return f"task:{task_id}:{field}"


def write(task_id: str, field: str, value: Any):
    """Store a value under task_id:field. Value is JSON-serialised."""
    serialised = json.dumps(value)
    if _REDIS_AVAILABLE:
        _client.setex(_key(task_id, field), TTL_SECONDS, serialised)
    else:
        _fallback[_key(task_id, field)] = serialised


def read(task_id: str, field: str, default: Any = None) -> Any:
    """Read a value. Returns default if not found."""
    if _REDIS_AVAILABLE:
        raw = _client.get(_key(task_id, field))
    else:
        raw = _fallback.get(_key(task_id, field))
    if raw is None:
        return default
    return json.loads(raw)


def append_to_list(task_id: str, field: str, item: Any):
    """Append an item to a stored list."""
    existing = read(task_id, field, default=[])
    if not isinstance(existing, list):
        existing = []
    existing.append(item)
    write(task_id, field, existing)


def get_all(task_id: str) -> Dict[str, Any]:
    """Return all fields stored for a task."""
    if _REDIS_AVAILABLE:
        pattern = f"task:{task_id}:*"
        keys = _client.keys(pattern)
        result = {}
        for k in keys:
            field = k.split(":", 2)[-1]
            raw = _client.get(k)
            result[field] = json.loads(raw) if raw else None
        return result
    else:
        prefix = f"task:{task_id}:"
        return {
            k[len(prefix):]: json.loads(v)
            for k, v in _fallback.items()
            if k.startswith(prefix)
        }


def clear(task_id: str):
    """Wipe all working memory for a completed task."""
    if _REDIS_AVAILABLE:
        pattern = f"task:{task_id}:*"
        keys = _client.keys(pattern)
        if keys:
            _client.delete(*keys)
    else:
        prefix = f"task:{task_id}:"
        for k in list(_fallback.keys()):
            if k.startswith(prefix):
                del _fallback[k]


def is_available() -> bool:
    return _REDIS_AVAILABLE
