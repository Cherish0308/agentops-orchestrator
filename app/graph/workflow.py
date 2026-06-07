from langgraph.graph import END, StateGraph

from app.agents.data import data_agent_node
from app.agents.escalation import escalation_node
from app.agents.research import research_agent_node
from app.agents.reviewer import reviewer_node
from app.agents.supervisor import supervisor_node
from app.agents.synthesizer import synthesizer_node
from app.agents.writer import writer_agent_node
from app.graph.state import AgentState

SPECIALIST_NODES = {
    "research_agent": "research_agent",
    "data_agent": "data_agent",
    "writer_agent": "writer_agent",
}

CONFIDENCE_THRESHOLD = 0.6
HIGH_RISK_LEVELS = {"high"}


def route_after_supervisor(state: AgentState) -> str:
    """
    After the supervisor produces a plan, decide:
    - escalate to human if confidence is low or risk is high
    - otherwise proceed to the first specialist
    """
    plan = state.get("execution_plan", {})
    confidence = plan.get("confidence_score", 1.0)
    risk = plan.get("risk_level", "low")

    needs_escalation = (confidence < CONFIDENCE_THRESHOLD) or (risk in HIGH_RISK_LEVELS)

    if needs_escalation:
        reason = (
            f"Confidence score {confidence:.2f} is below threshold {CONFIDENCE_THRESHOLD}."
            if confidence < CONFIDENCE_THRESHOLD
            else f"Risk level '{risk}' requires human approval."
        )
        state["escalation_required"] = True
        state["escalation_reason"] = reason
        return "human_escalation"

    return route_to_specialist(state)


def route_after_escalation(state: AgentState) -> str:
    """
    After a human reviews:
    - approve → proceed to first specialist
    - reject  → back to supervisor with feedback
    """
    decision = state.get("human_decision")
    if decision == "approve":
        return route_to_specialist(state)
    return "supervisor"


def route_to_specialist(state: AgentState) -> str:
    subtasks = state.get("execution_plan", {}).get("subtasks", [])
    idx = state.get("current_subtask_index", 0)

    if idx >= len(subtasks):
        return "reviewer"

    assigned_agent = subtasks[idx].get("assigned_agent", "writer_agent")
    return SPECIALIST_NODES.get(assigned_agent, "writer_agent")


def route_after_reviewer(state: AgentState) -> str:
    requires_rework = state.get("review_result", {}).get("requires_rework", False)
    rework_count = state.get("rework_count", 0)

    if requires_rework and rework_count < 1:
        return "supervisor"

    return "synthesizer"


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

    # Supervisor → escalation or specialist
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

    # Escalation → specialist (approved) or supervisor (rejected)
    graph.add_conditional_edges(
        "human_escalation",
        route_after_escalation,
        {
            "research_agent": "research_agent",
            "data_agent": "data_agent",
            "writer_agent": "writer_agent",
            "reviewer": "reviewer",
            "supervisor": "supervisor",
        },
    )

    # Each specialist → next specialist or reviewer
    for specialist in SPECIALIST_NODES.values():
        graph.add_conditional_edges(
            specialist,
            route_to_specialist,
            {
                "research_agent": "research_agent",
                "data_agent": "data_agent",
                "writer_agent": "writer_agent",
                "reviewer": "reviewer",
            },
        )

    # Reviewer → rework (supervisor) or synthesizer
    graph.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        {
            "supervisor": "supervisor",
            "synthesizer": "synthesizer",
        },
    )

    graph.add_edge("synthesizer", END)

    return graph.compile()
