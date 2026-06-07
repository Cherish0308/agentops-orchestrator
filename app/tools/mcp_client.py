"""
MCP (Model Context Protocol) tool integration.

MCP is Anthropic's open standard for connecting AI agents to external tools
and data sources via a JSON-RPC protocol over stdio or HTTP.

This module:
  1. Discovers tools from configured MCP servers
  2. Wraps each discovered tool as a callable that agents can use
  3. Registers them in the tool registry with appropriate agent permissions
  4. Handles the MCP request/response cycle

MCP server config is read from MCP_SERVERS env var (JSON):
  MCP_SERVERS='[{"name":"filesystem","command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","/tmp"]}]'

Falls back gracefully if no MCP servers are configured.
"""
import json
import os
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional


MCP_SERVERS_CONFIG = os.getenv("MCP_SERVERS", "[]")


class MCPServer:
    """
    Manages a connection to a single MCP server process over stdio.
    Sends JSON-RPC requests and reads responses.
    """

    def __init__(self, name: str, command: str, args: List[str]):
        self.name = name
        self.command = command
        self.args = args
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._request_id = 0

    def start(self) -> bool:
        try:
            self._process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            # Initialize the MCP session
            self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "agentops", "version": "1.0"},
            })
            return True
        except Exception:
            return False

    def stop(self):
        if self._process:
            self._process.terminate()
            self._process = None

    def list_tools(self) -> List[Dict[str, Any]]:
        response = self._send_request("tools/list", {})
        return response.get("result", {}).get("tools", [])

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        response = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        result = response.get("result", {})
        content = result.get("content", [])
        # Extract text content from MCP response
        texts = [c["text"] for c in content if c.get("type") == "text"]
        return "\n".join(texts) if texts else result

    def _send_request(self, method: str, params: Dict) -> Dict:
        with self._lock:
            if not self._process:
                return {}
            self._request_id += 1
            request = json.dumps({
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params,
            })
            try:
                self._process.stdin.write(request + "\n")
                self._process.stdin.flush()
                # Read response with timeout
                line = self._read_line(timeout=10)
                return json.loads(line) if line else {}
            except Exception:
                return {}

    def _read_line(self, timeout: float = 10) -> Optional[str]:
        """Read one line from stdout with a timeout."""
        result = [None]

        def reader():
            try:
                result[0] = self._process.stdout.readline()
            except Exception:
                pass

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        t.join(timeout=timeout)
        return result[0]


# ── MCP Tool Registry integration ─────────────────────────────────────────────

def _make_tool_fn(server: MCPServer, tool_name: str):
    """Create a callable wrapper for an MCP tool."""
    def call(**kwargs) -> Any:
        return server.call_tool(tool_name, kwargs)
    call.__name__ = f"mcp_{server.name}_{tool_name}"
    return call


def load_mcp_tools() -> List[Dict[str, Any]]:
    """
    Start configured MCP servers, discover their tools,
    and return a list of tool definitions ready to register.

    Returns:
        List of dicts: {name, description, input_schema, fn, server_name}
    """
    try:
        configs = json.loads(MCP_SERVERS_CONFIG)
    except json.JSONDecodeError:
        configs = []

    discovered = []

    for cfg in configs:
        server = MCPServer(
            name=cfg.get("name", "mcp"),
            command=cfg.get("command", ""),
            args=cfg.get("args", []),
        )
        if not server.start():
            continue

        for tool in server.list_tools():
            tool_name = tool.get("name", "")
            if not tool_name:
                continue

            input_schema = tool.get("inputSchema", {}).get("properties", {})

            discovered.append({
                "name": f"mcp_{server.name}_{tool_name}",
                "description": f"[MCP:{server.name}] {tool.get('description', '')}",
                "input_schema": {k: v.get("type", "str") for k, v in input_schema.items()},
                "fn": _make_tool_fn(server, tool_name),
                "server_name": server.name,
            })

    return discovered


def register_mcp_tools(allowed_agents: Optional[List[str]] = None):
    """
    Discover MCP tools and register them in the tool registry.
    Called once at startup.
    """
    from app.tools.registry import registry, ToolDefinition

    if allowed_agents is None:
        allowed_agents = ["research_agent", "data_agent", "writer_agent"]

    tools = load_mcp_tools()

    for t in tools:
        registry.register(ToolDefinition(
            name=t["name"],
            description=t["description"],
            input_schema=t["input_schema"],
            output_schema={"result": "any"},
            allowed_agents=allowed_agents,
            fn=t["fn"],
            rate_limit_per_minute=20,
        ))

    return len(tools)
