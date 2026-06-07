import os

# All file operations are sandboxed to this directory
WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "workspace")


def _safe_path(filename: str) -> str:
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    # Prevent path traversal attacks
    safe = os.path.normpath(os.path.join(WORKSPACE_DIR, filename))
    if not safe.startswith(os.path.abspath(WORKSPACE_DIR)):
        raise ValueError(f"Access outside workspace is not allowed: {filename}")
    return safe


def file_read(filename: str) -> dict:
    path = _safe_path(filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found in workspace: {filename}")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"filename": filename, "content": content, "size_bytes": len(content.encode())}


def file_write(filename: str, content: str) -> dict:
    path = _safe_path(filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"filename": filename, "written_bytes": len(content.encode()), "path": path}
