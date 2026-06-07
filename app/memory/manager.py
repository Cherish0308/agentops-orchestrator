"""
Memory Manager.

Sits between agents and the raw memory stores.
Responsibilities:
  - extract_and_save()  : called after task completion — pulls key facts
                          from the task result and saves them to long-term memory
  - consolidate()       : merges near-duplicate memories into summaries
  - expire_stale()      : removes memories past their TTL
  - importance_score()  : compute a score based on recency + access frequency
"""
import time
from typing import Any, Dict, List

from app.llm import call_llm
from app.memory import long_term


# ── Extraction ─────────────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = """
You are a memory extraction agent. Given a completed AI task, extract key facts worth remembering.

Return ONLY valid JSON with this schema:
{
  "task_outcome": "1-2 sentence summary of what was accomplished",
  "approach_that_worked": "which agents and tools were most useful, and why",
  "user_preference": "any preference the user revealed (tone, format, depth)",
  "domain_facts": ["fact 1", "fact 2"]
}

If you cannot determine a field, use null.
"""


def extract_and_save(
    user_id: str,
    task_id: str,
    original_request: str,
    execution_plan: Dict[str, Any],
    agent_outputs: Dict[str, Any],
    final_output: str,
    tools_used: List[str],
    success: bool = True,
) -> List[str]:
    """
    After a task finishes, extract memories and store them.
    Returns a list of saved memory IDs.
    """
    import json

    user_prompt = f"""
Original request: {original_request}

Execution plan summary: {execution_plan.get('task_summary', '')}
Tools used: {', '.join(tools_used) or 'none'}
Success: {success}

Final output (first 500 chars): {str(final_output)[:500]}
"""

    raw = call_llm(EXTRACTION_SYSTEM, user_prompt)
    saved_ids = []

    try:
        extracted = json.loads(raw)
    except Exception:
        extracted = {}

    # 1. Task outcome
    if extracted.get("task_outcome"):
        mid = long_term.save(
            user_id=user_id,
            content=f"Task: {original_request}\nOutcome: {extracted['task_outcome']}",
            memory_type="task_outcome",
            task_id=task_id,
            tools_used=tools_used,
            success=success,
            importance=0.6,
        )
        saved_ids.append(mid)

    # 2. Approach
    if extracted.get("approach_that_worked"):
        mid = long_term.save(
            user_id=user_id,
            content=extracted["approach_that_worked"],
            memory_type="approach",
            task_id=task_id,
            tools_used=tools_used,
            success=success,
            importance=0.7,
        )
        saved_ids.append(mid)

    # 3. User preference
    if extracted.get("user_preference"):
        mid = long_term.save(
            user_id=user_id,
            content=extracted["user_preference"],
            memory_type="user_preference",
            task_id=task_id,
            importance=0.8,
        )
        saved_ids.append(mid)

    # 4. Domain facts
    for fact in (extracted.get("domain_facts") or []):
        if fact:
            mid = long_term.save(
                user_id=user_id,
                content=fact,
                memory_type="domain_fact",
                task_id=task_id,
                importance=0.5,
            )
            saved_ids.append(mid)

    return saved_ids


# ── Retrieval for planning ─────────────────────────────────────────────────────

def retrieve_for_planning(user_id: str, request: str) -> str:
    """
    Query long-term memory and format the results as a context block
    to inject into the supervisor's planning prompt.
    Returns an empty string if no relevant memories exist.
    """
    memories = long_term.query(user_id=user_id, query_text=request, n_results=5)

    if not memories:
        return ""

    lines = ["=== Relevant past experience ==="]
    for m in memories:
        tag = f"[{m['type']} | similarity={m['similarity']} | success={m['success']}]"
        lines.append(f"{tag}\n{m['content']}")
    lines.append("=== End of memory context ===")

    return "\n\n".join(lines)


# ── Importance scoring ─────────────────────────────────────────────────────────

def importance_score(
    base: float,
    access_count: int,
    created_at: float,
    last_accessed_at: float,
) -> float:
    """
    Score = base + frequency_boost - recency_decay
    Clamped to [0, 1].
    """
    frequency_boost = min(0.2, access_count * 0.01)
    days_since_access = (time.time() - last_accessed_at) / 86400
    recency_decay = min(0.3, days_since_access * 0.005)
    return max(0.0, min(1.0, base + frequency_boost - recency_decay))


# ── Consolidation ──────────────────────────────────────────────────────────────

def consolidate_user_memories(user_id: str) -> int:
    """
    Find memories of the same type with high similarity and merge them.
    Returns the number of memories removed.
    """
    import json

    all_mems = long_term.get_user_memories(user_id, limit=200)
    if len(all_mems) < 10:
        return 0  # not enough to bother

    # Group by type
    by_type: Dict[str, List[Dict]] = {}
    for m in all_mems:
        by_type.setdefault(m.get("type", "unknown"), []).append(m)

    removed = 0
    for mem_type, mems in by_type.items():
        if len(mems) < 3:
            continue

        contents = "\n".join(f"- {m['content']}" for m in mems[:10])
        summary_prompt = f"""
These are {mem_type} memories. Merge them into a single concise summary (2-4 sentences).
Memories:
{contents}
Return only the summary text, nothing else.
"""
        summary = call_llm("You are a memory consolidation agent.", summary_prompt)

        # Save consolidated memory
        long_term.save(
            user_id=user_id,
            content=summary,
            memory_type=mem_type,
            importance=0.9,
            extra_metadata={"consolidated": "true", "source_count": str(len(mems[:10]))},
        )

        # Delete the originals
        for m in mems[:10]:
            long_term.delete_memory(m["id"])
            removed += 1

    return removed


# ── Expiry ─────────────────────────────────────────────────────────────────────

def expire_stale_memories(user_id: str) -> int:
    """
    Manually trigger expiry check for a user.
    ChromaDB queries already skip expired docs, but this physically removes them.
    """
    now = time.time()
    all_mems = long_term.get_user_memories(user_id, limit=500)
    removed = 0
    for m in all_mems:
        expires = float(m.get("expires_at", float("inf")))
        importance = float(m.get("importance_score", 0.5))
        # Also remove very low importance memories older than 7 days
        age_days = (now - float(m.get("created_at", now))) / 86400
        if expires < now or (importance < 0.2 and age_days > 7):
            long_term.delete_memory(m["id"])
            removed += 1
    return removed
