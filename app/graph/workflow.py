from langgraph.graph import END, StateGraph

from app.agents.reviewer import reviewer_node
from app.agents.specialist import specialist_execution_node
from app.agents.supervisor import supervisor_node
from app.agents.synthesizer import synthesizer_node
from app.graph.state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("specialist", specialist_execution_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "specialist")
    graph.add_edge("specialist", "reviewer")
    graph.add_edge("reviewer", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()
