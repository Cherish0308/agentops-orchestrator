import json
from typing import Any, Dict

import app.tools  # ensure tools are registered
from app.tools.registry import registry, tool_log
from app.llm import call_llm, current_task_id, current_node
from app.human_review import detect_sensitive_operations, determine_approval_level
from app.observability.tracer import tracer

AGENT_ROLES = {
    "research_agent": (
        "You are the Research Agent. Gather information using the tools available to you, "
        "analyze sources, and summarize findings relevant to the subtask."
    ),
    "data_agent": (
        "You are the Data Agent. Process data, run code, extract insights, "
        "and structure information for downstream use."
    ),
    "writer_agent": (
        "You are the Writer Agent. Draft clear, polished content "
        "tailored to the end user. You may read/write files as needed."
    ),
}

MAX_TOOL_ROUNDS = 5
MAX_SUBTASK_FAILURES = 2   # escalate after this many failures on the same subtask


def _build_tool_selection_prompt(agent_name: str, subtask: Dict, available_tools: list, prior_outputs: Dict) -> str:
    return f"""
You are deciding whether to call a tool to complete this subtask.

Available tools:
{json.dumps(available_tools, indent=2)}

Subtask:
{json.dumps(subtask, indent=2)}

Prior agent outputs:
{json.dumps(prior_outputs, indent=2)}

If you need a tool, respond with ONLY valid JSON:
{{
  "use_tool": true,
  "tool_name": "<name>",
  "inputs": {{ <matching the tool input_schema> }}
}}

If you have enough information and do NOT need a tool, respond with:
{{
  "use_tool": false
}}
"""


def _build_final_output_prompt(agent_name: str, role_prompt: str, subtask: Dict, prior_outputs: Dict, tool_results: list) -> tuple[str, str]:
    system = f"""
{role_prompt}
Execute the assigned subtask using the tool results provided and return ONLY valid JSON.

JSON schema:
{{
  "subtask_id": "id",
  "assigned_agent": "{agent_name}",
  "output": "result content",
  "confidence_score": 0.0,
  "tools_used": ["tool names"],
  "errors": []
}}
"""
    user = f"""
Subtask:
{json.dumps(subtask, indent=2)}

Tool results gathered:
{json.dumps(tool_results, indent=2)}

Prior agent outputs:
{json.dumps(prior_outputs, indent=2)}

Now produce the final subtask output.
"""
    return system, user


def run_specialist_node(state: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
    subtasks = state.get("execution_plan", {}).get("subtasks", [])
    idx = state.get("current_subtask_index", 0)

    if idx >= len(subtasks):
        return {}

    subtask = subtasks[idx]
    subtask_id = subtask.get("id", f"subtask_{idx + 1}")
    role_prompt = AGENT_ROLES[agent_name]
    prior_outputs = dict(state.get("agent_outputs", {}))
    available_tools = registry.list_for_agent(agent_name)

    # Set observability context so call_llm can tag its traces
    task_id = state.get("task_id", "")
    current_task_id.set(task_id)
    current_node.set(agent_name)
    tracer.node_enter(task_id, agent_name, input_keys=["subtask", "prior_outputs"])

    # ── Sensitive operation detection ──────────────────────────────────────────
    subtask_text = f"{subtask.get('description', '')} {subtask.get('expected_output', '')}"
    sensitive_ops = detect_sensitive_operations(subtask_text)

    if sensitive_ops:
        level, reason = determine_approval_level(sensitive_ops=sensitive_ops)
        tracer.node_exit(task_id, agent_name)
        return {
            "escalation_required": True,
            "escalation_reason": reason,
            "approval_level": level,
        }

    # ── Failure count check ────────────────────────────────────────────────────
    failure_counts: Dict[str, int] = dict(state.get("subtask_failure_counts", {}))
    if failure_counts.get(subtask_id, 0) >= MAX_SUBTASK_FAILURES:
        level, reason = determine_approval_level(specialist_failure_count=failure_counts[subtask_id])
        tracer.node_exit(task_id, agent_name)
        return {
            "escalation_required": True,
            "escalation_reason": reason,
            "approval_level": level,
        }

    # ── Tool use loop ──────────────────────────────────────────────────────────
    tool_results = []
    tools_used = []

    for _ in range(MAX_TOOL_ROUNDS):
        if not available_tools:
            break

        decision_raw = call_llm(
            "You are a tool-selection assistant. Respond only with valid JSON.",
            _build_tool_selection_prompt(agent_name, subtask, available_tools, prior_outputs),
        )

        try:
            decision = json.loads(decision_raw)
        except json.JSONDecodeError:
            break

        if not decision.get("use_tool"):
            break

        tool_name = decision.get("tool_name")
        inputs = decision.get("inputs", {})

        result = registry.invoke(tool_name, agent_name, inputs)
        tool_results.append({"tool": tool_name, "inputs": inputs, "result": result})
        if tool_name not in tools_used:
            tools_used.append(tool_name)

        if not result.get("success"):
            break

    # ── Final synthesis ────────────────────────────────────────────────────────
    system_prompt, user_prompt = _build_final_output_prompt(
        agent_name, role_prompt, subtask, prior_outputs, tool_results
    )
    raw_response = call_llm(system_prompt, user_prompt)

    errors = []
    try:
        output = json.loads(raw_response)
        if not isinstance(output, dict):
            raise ValueError("Non-dict response")
    except (json.JSONDecodeError, ValueError):
        errors.append("JSON parsing failed on final output")
        output = {
            "subtask_id": subtask_id,
            "assigned_agent": agent_name,
            "output": raw_response,
            "confidence_score": 0.6,
            "tools_used": tools_used,
            "errors": errors,
        }

    # ── Track failures for next attempt ───────────────────────────────────────
    if errors or output.get("errors"):
        failure_counts[subtask_id] = failure_counts.get(subtask_id, 0) + 1
    else:
        failure_counts.pop(subtask_id, None)  # clear on success

    output["tools_used"] = tools_used
    output["tool_invocation_logs"] = tool_log.get_logs()

    # Trace tool calls
    for log_entry in tool_log.get_logs():
        tracer.tool_call(
            task_id=task_id,
            node=agent_name,
            tool_name=log_entry["tool"],
            inputs=log_entry["inputs"],
            output=log_entry["output"],
            latency_ms=log_entry["latency_ms"],
            success=log_entry["success"],
            error=log_entry.get("error"),
        )

    tracer.node_exit(task_id, agent_name, output_keys=["agent_outputs", "current_subtask_index"])

    prior_outputs[subtask_id] = output

    return {
        "agent_outputs": prior_outputs,
        "current_subtask_index": idx + 1,
        "subtask_failure_counts": failure_counts,
        "errors": [{"agent": agent_name, "subtask_id": subtask_id, "error": e} for e in errors],
    }
