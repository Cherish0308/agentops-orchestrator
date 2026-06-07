from app.tools.registry import registry, ToolDefinition
from app.tools.web_search import web_search
from app.tools.file_tools import file_read, file_write
from app.tools.code_executor import code_execute

registry.register(ToolDefinition(
    name="web_search",
    description="Search the web for information on a topic. Returns titles, URLs, and snippets.",
    input_schema={"query": "str", "max_results": "int (optional, default 5)"},
    output_schema={"query": "str", "results": "list of {title, url, snippet}"},
    allowed_agents=["research_agent", "data_agent"],
    fn=web_search,
    rate_limit_per_minute=20,
))

registry.register(ToolDefinition(
    name="file_read",
    description="Read a file from the shared workspace directory.",
    input_schema={"filename": "str"},
    output_schema={"filename": "str", "content": "str", "size_bytes": "int"},
    allowed_agents=["research_agent", "data_agent", "writer_agent"],
    fn=file_read,
    rate_limit_per_minute=60,
))

registry.register(ToolDefinition(
    name="file_write",
    description="Write content to a file in the shared workspace directory.",
    input_schema={"filename": "str", "content": "str"},
    output_schema={"filename": "str", "written_bytes": "int"},
    allowed_agents=["research_agent", "data_agent", "writer_agent"],
    fn=file_write,
    rate_limit_per_minute=60,
))

registry.register(ToolDefinition(
    name="code_execute",
    description="Execute a Python code snippet in a sandboxed subprocess. Returns stdout, stderr, and return code.",
    input_schema={"code": "str", "timeout_seconds": "int (optional, default 10)"},
    output_schema={"stdout": "str", "stderr": "str", "returncode": "int", "success": "bool"},
    allowed_agents=["data_agent"],
    fn=code_execute,
    rate_limit_per_minute=10,
))
