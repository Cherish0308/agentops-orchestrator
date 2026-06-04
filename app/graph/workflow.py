from app.graph.state import AgentState
from app.agents.supervisor import supervisor_node
from app.agents.specialist import specialist_execution_node
from app.agents.reviewer import reviewer_node
from app.agents.synthesizer import synthesizer_node


class SimpleWorkflow:
    def __init__(self):
        self.nodes = {
            "supervisor": supervisor_node,
            "specialist": specialist_execution_node,
            "reviewer": reviewer_node,
            "synthesizer": synthesizer_node
        }

    def invoke(self, initial_state):
        state = initial_state.copy()

        # Run supervisor
        result = self.nodes["supervisor"](state)
        state.update(result)

        # Run specialist
        result = self.nodes["specialist"](state)
        state.update(result)

        # Run reviewer
        result = self.nodes["reviewer"](state)
        state.update(result)

        # Run synthesizer
        result = self.nodes["synthesizer"](state)
        state.update(result)

        return state


def build_graph():
    return SimpleWorkflow()
