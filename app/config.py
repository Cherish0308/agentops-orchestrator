import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM providers ──────────────────────────────────────────────────────────────
NVIDIA_API_KEY    = os.getenv("NVIDIA_API_KEY")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")    # fallback if NVIDIA key missing
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") # fallback if NVIDIA key missing
LLM_MODEL         = os.getenv("LLM_MODEL", "meta/llama-3.1-70b-instruct")

# NVIDIA NIM endpoint (OpenAI-compatible)
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# ── Redis ──────────────────────────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_URL  = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

# ── PostgreSQL ─────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://agentops:agentops@localhost:5432/agentops"
)

# ── ChromaDB ───────────────────────────────────────────────────────────────────
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
