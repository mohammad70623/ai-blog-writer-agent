from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from app.models.state import State
from app.agents.nodes.router import router_node, route_next
from app.agents.nodes.research import research_node
from app.agents.nodes.worker import orchestrator_node, fanout, worker_node
from app.agents.nodes.reducer import merge_content, decide_images, generate_and_place_images


# ──────────────────────────────────────────────
# Reducer subgraph
# merge_content → decide_images → generate_and_place_images
# ──────────────────────────────────────────────

def build_reducer_subgraph():
    g = StateGraph(State)
    g.add_node("merge_content", merge_content)
    g.add_node("decide_images", decide_images)
    g.add_node("generate_and_place_images", generate_and_place_images)

    g.add_edge(START, "merge_content")
    g.add_edge("merge_content", "decide_images")
    g.add_edge("decide_images", "generate_and_place_images")
    g.add_edge("generate_and_place_images", END)

    return g.compile()


# ──────────────────────────────────────────────
# Main graph
# router → (research?) → orchestrator → workers → reducer
# ──────────────────────────────────────────────

def build_graph():
    reducer_subgraph = build_reducer_subgraph()

    g = StateGraph(State)
    g.add_node("router", router_node)
    g.add_node("research", research_node)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("worker", worker_node)
    g.add_node("reducer", reducer_subgraph)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        route_next,
        {"research": "research", "orchestrator": "orchestrator"},
    )
    g.add_edge("research", "orchestrator")
    g.add_conditional_edges("orchestrator", fanout, ["worker"])
    g.add_edge("worker", "reducer")
    g.add_edge("reducer", END)

    return g.compile()


# Compiled app — import this everywhere
app = build_graph()
