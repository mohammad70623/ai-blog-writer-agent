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


# ──────────────────────────────────────────────
# Worker — writes one section
# ──────────────────────────────────────────────

WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Hit the target word count ±15%.
- Output ONLY the section markdown, starting with "## <Section Title>".

Scope guard:
- If blog_kind=="news_roundup": focus on events and implications only.
  Do NOT drift into how-to tutorials.

Grounding:
- open_book mode: do not state any specific event/company/model/funding/policy
  claim unless it is supported by the provided Evidence URLs.
  Attach a Markdown link ([Source](URL)) for each supported claim.
  For unsupported claims, write "Not found in provided sources."
- hybrid mode with requires_citations=True: cite Evidence URLs for external claims.

Code:
- If requires_code=True, include at least one minimal, runnable code snippet.
"""


def worker_node(payload: dict) -> dict:
    llm = get_llm()

    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
        for e in evidence[:20]
    )

    section_md: str = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {payload['topic']}\n"
                    f"Mode: {payload.get('mode')}\n"
                    f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY cite these URLs):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}
