"""LangGraph pipeline — wires all agent nodes with conditional routing."""
from langgraph.graph import StateGraph, END
from agents.state import PipelineState
from agents.validation_node import validation_node
from agents.profiling_node import profiling_node
from agents.quality_node import quality_node
from agents.anomaly_node import anomaly_node
from agents.score_node import score_node
from agents.insight_node import insight_node
from agents.cleaning_node import cleaning_node


def _route_after_validation(state: PipelineState) -> str:
    return "failed" if state.get("status") == "failed" else "profiling"


def _route_after_insight(state: PipelineState) -> str:
    """After insights, pause at AWAITING_DECISION — graph waits for human input."""
    return END  # FastAPI resumes the graph via cleaning_node when user decides


def _route_after_cleaning(state: PipelineState) -> str:
    return "failed" if state.get("status") == "failed" else END


def _failed_node(state: PipelineState) -> PipelineState:
    return {**state, "status": "failed"}


def build_analysis_graph() -> StateGraph:
    """Graph for the analysis pipeline (upload → awaiting_decision)."""
    g = StateGraph(PipelineState)

    g.add_node("validation", validation_node)
    g.add_node("profiling",  profiling_node)
    g.add_node("quality",    quality_node)
    g.add_node("anomaly",    anomaly_node)
    g.add_node("score",      score_node)
    g.add_node("insight",    insight_node)
    g.add_node("failed",     _failed_node)

    g.set_entry_point("validation")

    g.add_conditional_edges("validation", _route_after_validation, {
        "profiling": "profiling",
        "failed":    "failed",
    })
    g.add_edge("profiling", "quality")
    g.add_edge("quality",   "anomaly")
    g.add_edge("anomaly",   "score")
    g.add_edge("score",     "insight")
    g.add_edge("insight",   END)
    g.add_edge("failed",    END)

    return g.compile()


def build_cleaning_graph() -> StateGraph:
    """Separate graph for the cleaning step only."""
    g = StateGraph(PipelineState)
    g.add_node("cleaning", cleaning_node)
    g.add_node("failed",   _failed_node)
    g.set_entry_point("cleaning")
    g.add_conditional_edges("cleaning", _route_after_cleaning, {
        END:      END,
        "failed": "failed",
    })
    g.add_edge("failed", END)
    return g.compile()


# Compiled singletons — import these in the API layer
analysis_graph = build_analysis_graph()
cleaning_graph = build_cleaning_graph()
