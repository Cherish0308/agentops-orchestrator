import time
from typing import Any, Callable, Dict, List, Optional


class ToolDefinition:
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        allowed_agents: List[str],
        fn: Callable,
        rate_limit_per_minute: int = 30,
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.allowed_agents = allowed_agents
        self.fn = fn
        self.rate_limit_per_minute = rate_limit_per_minute
        self._call_timestamps: List[float] = []

    def _check_rate_limit(self):
        now = time.time()
        self._call_timestamps = [t for t in self._call_timestamps if now - t < 60]
        if len(self._call_timestamps) >= self.rate_limit_per_minute:
            raise RuntimeError(f"Tool '{self.name}' rate limit exceeded ({self.rate_limit_per_minute}/min)")
        self._call_timestamps.append(now)


class ToolInvocationLog:
    def __init__(self):
        self.entries: List[Dict[str, Any]] = []

    def record(self, tool_name: str, agent: str, inputs: Dict, output: Any, latency_ms: float, success: bool, error: Optional[str] = None):
        self.entries.append({
            "tool": tool_name,
            "agent": agent,
            "inputs": inputs,
            "output": output,
            "latency_ms": round(latency_ms, 2),
            "success": success,
            "error": error,
            "timestamp": time.time(),
        })

    def get_logs(self) -> List[Dict[str, Any]]:
        return self.entries


# Global log — shared across the request lifecycle
tool_log = ToolInvocationLog()


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_for_agent(self, agent_name: str) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
            if agent_name in t.allowed_agents
        ]

    def invoke(self, tool_name: str, agent_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found in registry")
        if agent_name not in tool.allowed_agents:
            raise PermissionError(f"Agent '{agent_name}' is not allowed to use tool '{tool_name}'")

        tool._check_rate_limit()

        start = time.time()
        error = None
        output = None
        success = False

        try:
            output = tool.fn(**inputs)
            success = True
        except Exception as e:
            error = str(e)
            output = {"error": error}
        finally:
            latency_ms = (time.time() - start) * 1000
            tool_log.record(tool_name, agent_name, inputs, output, latency_ms, success, error)

        return {"success": success, "output": output, "error": error}


# Singleton registry
registry = ToolRegistry()
