from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage
from app.models.state import State, RouterDecision
from app.agents.llm import get_llm

ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts, well-known topics.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy topics.

If needs_research=true:
- Output 3–10 high-signal, scoped search queries.
- For open_book weekly roundup, include queries reflecting the last 7 days.
"""


def router_node(state: State) -> dict:
    llm = get_llm()
    decider = llm.with_structured_output(RouterDecision)

    decision: RouterDecision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ]
    )

    recency_map = {
        "open_book": 7,
        "hybrid": 45,
        "closed_book": 3650,
    }

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_map.get(decision.mode, 3650),
    }


def route_next(state: State) -> str:
    """Conditional edge: go to research or skip straight to orchestrator."""
    return "research" if state["needs_research"] else "orchestrator"
