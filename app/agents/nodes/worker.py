from __future__ import annotations
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Send
from app.models.state import State, Plan, Task, EvidenceItem
from app.agents.llm import get_llm

# ──────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────

ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words (120–550).
- Tags are flexible; do not force a fixed taxonomy.

Grounding rules by mode:
- closed_book: evergreen content, no evidence needed.
- hybrid: use evidence for up-to-date examples; mark those tasks
  requires_research=True and requires_citations=True.
- open_book: news roundup style (blog_kind="news_roundup").
  No tutorial content unless specifically requested.
  If evidence is weak, reflect that honestly — do not invent events.

Output must match the Plan schema exactly.
"""


def orchestrator_node(state: State) -> dict:
    llm = get_llm()
    planner = llm.with_structured_output(Plan)

    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])
    forced_kind = "news_roundup" if mode == "open_book" else None

    plan: Plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force blog_kind=news_roundup. ' if forced_kind else ''}\n\n"
                    f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )

    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {"plan": plan}


# ──────────────────────────────────────────────
# Fanout — sends one message per task to worker
# ──────────────────────────────────────────────

def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]


