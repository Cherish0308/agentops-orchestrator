"""
Long-Term Semantic Memory — ChromaDB-backed vector store.

After every task completes, key facts are extracted and embedded here.
When the supervisor plans a new task, it queries this store to find
similar past experiences and inject them into the planning prompt.

Memory document structure:
{
    "id":           unique memory id
    "user_id":      scopes memories per user
    "type":         "task_outcome" | "approach" | "user_preference" | "domain_fact"
    "content":      the actual text that gets embedded + retrieved
    "metadata":     task_id, tools_used, success, importance_score, access_count,
                    created_at, last_accessed_at, expires_at
}

Falls back gracefully to an in-process list if ChromaDB is unavailable.
"""
import os
import time
import uuid
from typing import Any, Dict, List, Optional

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
COLLECTION_NAME = "agent_memory"
DEFAULT_IMPORTANCE = 0.5
MEMORY_TTL_DAYS = int(os.getenv("MEMORY_TTL_DAYS", 90))

try:
    import chromadb
    from chromadb.config import Settings

    _chroma = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )
    _chroma.heartbeat()
    _collection = _chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    _CHROMA_AVAILABLE = True
except Exception:
    _CHROMA_AVAILABLE = False
    _fallback_store: List[Dict[str, Any]] = []


def _expires_at() -> float:
    return time.time() + MEMORY_TTL_DAYS * 86400


def save(
    user_id: str,
    content: str,
    memory_type: str,
    task_id: str = "",
    tools_used: List[str] = [],
    success: bool = True,
    importance: float = DEFAULT_IMPORTANCE,
    extra_metadata: Dict[str, Any] = {},
) -> str:
    """Embed and store a memory. Returns the memory id."""
    memory_id = str(uuid.uuid4())
    now = time.time()

    metadata = {
        "user_id": user_id,
        "type": memory_type,
        "task_id": task_id,
        "tools_used": ",".join(tools_used),
        "success": str(success),
        "importance_score": importance,
        "access_count": 0,
        "created_at": now,
        "last_accessed_at": now,
        "expires_at": _expires_at(),
        **{k: str(v) for k, v in extra_metadata.items()},
    }

    if _CHROMA_AVAILABLE:
        _collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas=[metadata],
        )
    else:
        _fallback_store.append({
            "id": memory_id,
            "content": content,
            "metadata": metadata,
        })

    return memory_id


def query(
    user_id: str,
    query_text: str,
    n_results: int = 5,
    memory_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve the most semantically similar memories for a user.
    Filters out expired memories and updates access metadata.
    """
    now = time.time()

    if _CHROMA_AVAILABLE:
        where: Dict[str, Any] = {"user_id": {"$eq": user_id}}
        if memory_type:
            where = {
                "$and": [
                    {"user_id": {"$eq": user_id}},
                    {"type": {"$eq": memory_type}},
                ]
            }

        try:
            results = _collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        memories = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for mem_id, doc, meta, dist in zip(ids, docs, metas, dists):
            # Skip expired memories
            if float(meta.get("expires_at", 0)) < now:
                continue

            # Boost importance on access
            new_count = int(meta.get("access_count", 0)) + 1
            new_importance = min(1.0, float(meta.get("importance_score", 0.5)) + 0.02)
            _collection.update(
                ids=[mem_id],
                metadatas=[{
                    **meta,
                    "access_count": new_count,
                    "importance_score": new_importance,
                    "last_accessed_at": now,
                }],
            )

            memories.append({
                "id": mem_id,
                "content": doc,
                "similarity": round(1 - dist, 4),
                "type": meta.get("type"),
                "task_id": meta.get("task_id"),
                "tools_used": meta.get("tools_used", "").split(","),
                "success": meta.get("success") == "True",
                "importance_score": new_importance,
                "access_count": new_count,
                "created_at": meta.get("created_at"),
            })

        return memories

    else:
        # Fallback: simple keyword overlap scoring
        scored = []
        query_words = set(query_text.lower().split())
        for entry in _fallback_store:
            meta = entry["metadata"]
            if meta.get("user_id") != user_id:
                continue
            if float(meta.get("expires_at", 0)) < now:
                continue
            if memory_type and meta.get("type") != memory_type:
                continue
            content_words = set(entry["content"].lower().split())
            overlap = len(query_words & content_words) / max(len(query_words), 1)
            scored.append((overlap, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": e["id"],
                "content": e["content"],
                "similarity": round(s, 4),
                "type": e["metadata"].get("type"),
                "task_id": e["metadata"].get("task_id"),
                "tools_used": e["metadata"].get("tools_used", "").split(","),
                "success": e["metadata"].get("success") == "True",
                "importance_score": float(e["metadata"].get("importance_score", 0.5)),
                "access_count": int(e["metadata"].get("access_count", 0)),
                "created_at": e["metadata"].get("created_at"),
            }
            for s, e in scored[:n_results]
        ]


def get_user_memories(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Return all memories for a user (for the memory dashboard)."""
    now = time.time()
    if _CHROMA_AVAILABLE:
        try:
            results = _collection.get(
                where={"user_id": {"$eq": user_id}},
                include=["documents", "metadatas"],
                limit=limit,
            )
            memories = []
            for mem_id, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
                if float(meta.get("expires_at", 0)) < now:
                    continue
                memories.append({"id": mem_id, "content": doc, **meta})
            return sorted(memories, key=lambda m: float(m.get("importance_score", 0)), reverse=True)
        except Exception:
            return []
    else:
        return [
            {"id": e["id"], "content": e["content"], **e["metadata"]}
            for e in _fallback_store
            if e["metadata"].get("user_id") == user_id
            and float(e["metadata"].get("expires_at", 0)) >= now
        ]


def delete_user_memories(user_id: str) -> int:
    """GDPR-style: delete all memories for a user. Returns count deleted."""
    if _CHROMA_AVAILABLE:
        try:
            results = _collection.get(where={"user_id": {"$eq": user_id}})
            ids = results["ids"]
            if ids:
                _collection.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0
    else:
        before = len(_fallback_store)
        _fallback_store[:] = [
            e for e in _fallback_store if e["metadata"].get("user_id") != user_id
        ]
        return before - len(_fallback_store)


def delete_memory(memory_id: str) -> bool:
    """Delete a single memory by id."""
    if _CHROMA_AVAILABLE:
        try:
            _collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False
    else:
        before = len(_fallback_store)
        _fallback_store[:] = [e for e in _fallback_store if e["id"] != memory_id]
        return len(_fallback_store) < before


def is_available() -> bool:
    return _CHROMA_AVAILABLE
