from langgraph.graph import END, StateGraph

from app.agents.data import data_agent_node
from app.agents.escalation import escalation_node
from app.agents.research import research_agent_node
from app.agents.reviewer import reviewer_node
from app.agents.supervisor import supervisor_node
from app.agents.synthesizer import synthesizer_node
from app.agents.writer import writer_agent_node
from app.graph.state import AgentState
from app.human_review import APPROVE_PLAN, NOTIFY, determine_approval_level

SPECIALIST_NODES = {
    "research_agent": "research_agent",
    "data_agent": "data_agent",
    "writer_agent": "writer_agent",
}

CONFIDENCE_THRESHOLD = 0.6
QUALITY_THRESHOLD = 0.5


# ── Routing functions ──────────────────────────────────────────────────────────

def route_after_supervisor(state: AgentState) -> str:
    """
    Triggers: low confidence, high risk, or user explicitly requested review.
    """
    # Check if a specialist already flagged escalation (sensitive ops / failures)
    if state.get("escalation_required"):
        return "human_escalation"

    plan = state.get("execution_plan", {})
    confidence = plan.get("confidence_score", 1.0)
    risk = plan.get("risk_level", "low")
    user_requested = state.get("original_request", "").lower().strip().startswith("review:")

    level, reason = determine_approval_level(
        confidence_score=confidence,
        risk_level=risk,
        user_requested=user_requested,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    needs_pause = level in (APPROVE_PLAN,)

    if needs_pause or level != NOTIFY:
        state["escalation_required"] = True
        state["escalation_reason"] = reason
        state["approval_level"] = level
        if level == NOTIFY:
            # still route to escalation node but it will immediately pass through
            return "human_escalation"
        return "human_escalation"

    return route_to_specialist(state)


def route_after_escalation(state: AgentState) -> str:
    """
    approve / modify → proceed to specialists
    take_over        → skip to synthesizer (human provided the output)
    reject           → back to supervisor to re-plan
    """
    decision = state.get("human_decision")

    if decision == "take_over":
        return "synthesizer"
    if decision in ("approve", "modify"):
        return route_to_specialist(state)
    return "supervisor"


def route_to_specialist(state: AgentState) -> str:
    # If a specialist set escalation_required mid-run, intercept
    if state.get("escalation_required"):
        return "human_escalation"

    subtasks = state.get("execution_plan", {}).get("subtasks", [])
    idx = state.get("current_subtask_index", 0)

    if idx >= len(subtasks):
        return "reviewer"

    assigned_agent = subtasks[idx].get("assigned_agent", "writer_agent")
    return SPECIALIST_NODES.get(assigned_agent, "writer_agent")


def route_after_reviewer(state: AgentState) -> str:
    """
    Triggers: low quality score → escalate for approve_action.
    Otherwise: rework once if reviewer rejected, then synthesize.
    """
    review = state.get("review_result", {})
    quality_score = review.get("quality_score", 1.0)
    requires_rework = review.get("requires_rework", False)
    rework_count = state.get("rework_count", 0)

    # Escalate if quality is too low
    if quality_score < QUALITY_THRESHOLD:
        level, reason = determine_approval_level(quality_score=quality_score)
        state["escalation_required"] = True
        state["escalation_reason"] = reason
        state["approval_level"] = level
        return "human_escalation"

    if requires_rework and rework_count < 1:
        return "supervisor"

    return "synthesizer"


# ── Graph construction ─────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("human_escalation", escalation_node)
    graph.add_node("research_agent", research_agent_node)
    graph.add_node("data_agent", data_agent_node)
    graph.add_node("writer_agent", writer_agent_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "human_escalation": "human_escalation",
            "research_agent": "research_agent",
            "data_agent": "data_agent",
            "writer_agent": "writer_agent",
            "reviewer": "reviewer",
        },
    )

    graph.add_conditional_edges(
        "human_escalation",
        route_after_escalation,
        {
            "research_agent": "research_agent",
            "data_agent": "data_agent",
            "writer_agent": "writer_agent",
            "reviewer": "reviewer",
            "supervisor": "supervisor",
            "synthesizer": "synthesizer",
        },
    )

    for specialist in SPECIALIST_NODES.values():
        graph.add_conditional_edges(
            specialist,
            route_to_specialist,
            {
                "human_escalation": "human_escalation",
                "research_agent": "research_agent",
                "data_agent": "data_agent",
                "writer_agent": "writer_agent",
                "reviewer": "reviewer",
            },
        )

    graph.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        {
            "human_escalation": "human_escalation",
            "supervisor": "supervisor",
            "synthesizer": "synthesizer",
        },
    )

    graph.add_edge("synthesizer", END)

    return graph.compile()
