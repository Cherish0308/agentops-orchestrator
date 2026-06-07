"""
PostgreSQL — task audit log.

Stores a permanent record of every task run:
  - task_id, user_id, original request
  - status: pending | running | completed | failed
  - execution plan, final output, errors
  - timing: submitted_at, started_at, completed_at

Uses SQLAlchemy Core (no ORM) for simplicity.
Falls back silently if Postgres is unavailable.
"""
import json
import time
from typing import Any, Dict, Optional

from app.config import DATABASE_URL

_engine = None
_tasks_table = None
_POSTGRES_AVAILABLE = False

try:
    from sqlalchemy import (
        Column, Float, MetaData, String, Table, Text,
        create_engine, insert, select, update,
    )

    _engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args={"connect_timeout": 3})

    _meta = MetaData()
    _tasks_table = Table(
        "task_runs", _meta,
        Column("task_id",       String(64),  primary_key=True),
        Column("user_id",       String(256), nullable=False),
        Column("request",       Text,        nullable=False),
        Column("status",        String(32),  default="pending"),
        Column("execution_plan",Text,        nullable=True),
        Column("final_output",  Text,        nullable=True),
        Column("errors",        Text,        nullable=True),
        Column("submitted_at",  Float,       nullable=False),
        Column("started_at",    Float,       nullable=True),
        Column("completed_at",  Float,       nullable=True),
    )

    with _engine.connect() as conn:
        _meta.create_all(_engine)
    _POSTGRES_AVAILABLE = True

except Exception:
    _POSTGRES_AVAILABLE = False


# ── Public API ─────────────────────────────────────────────────────────────────

def record_submitted(task_id: str, user_id: str, request: str):
    if not _POSTGRES_AVAILABLE:
        return
    try:
        with _engine.begin() as conn:
            conn.execute(insert(_tasks_table).values(
                task_id=task_id,
                user_id=user_id,
                request=request,
                status="pending",
                submitted_at=time.time(),
            ))
    except Exception:
        pass


def record_started(task_id: str):
    _update_task(task_id, {"status": "running", "started_at": time.time()})


def record_completed(task_id: str, execution_plan: Dict, final_output: str, errors: list):
    _update_task(task_id, {
        "status": "completed",
        "execution_plan": json.dumps(execution_plan),
        "final_output": final_output,
        "errors": json.dumps(errors),
        "completed_at": time.time(),
    })


def record_failed(task_id: str, error: str):
    _update_task(task_id, {
        "status": "failed",
        "errors": json.dumps([error]),
        "completed_at": time.time(),
    })


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    if not _POSTGRES_AVAILABLE:
        return None
    try:
        with _engine.connect() as conn:
            row = conn.execute(
                select(_tasks_table).where(_tasks_table.c.task_id == task_id)
            ).fetchone()
        if not row:
            return None
        return dict(row._mapping)
    except Exception:
        return None


def get_user_tasks(user_id: str, limit: int = 20) -> list:
    if not _POSTGRES_AVAILABLE:
        return []
    try:
        from sqlalchemy import desc
        with _engine.connect() as conn:
            rows = conn.execute(
                select(_tasks_table)
                .where(_tasks_table.c.user_id == user_id)
                .order_by(desc(_tasks_table.c.submitted_at))
                .limit(limit)
            ).fetchall()
        return [dict(r._mapping) for r in rows]
    except Exception:
        return []


def is_available() -> bool:
    return _POSTGRES_AVAILABLE


def _update_task(task_id: str, values: Dict):
    if not _POSTGRES_AVAILABLE:
        return
    try:
        with _engine.begin() as conn:
            conn.execute(
                update(_tasks_table)
                .where(_tasks_table.c.task_id == task_id)
                .values(**values)
            )
    except Exception:
        pass
