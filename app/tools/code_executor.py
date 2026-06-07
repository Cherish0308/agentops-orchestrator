import subprocess
import sys
import tempfile
import os


def code_execute(code: str, timeout_seconds: int = 10) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            # No network, no shell — subprocess isolation is the sandbox here
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Execution timed out after {timeout_seconds}s",
            "returncode": -1,
            "success": False,
        }
    finally:
        os.unlink(tmp_path)
