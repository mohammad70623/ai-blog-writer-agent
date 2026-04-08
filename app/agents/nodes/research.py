from __future__ import annotations
from datetime import date, timedelta
from typing import List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from app.models.state import State, EvidenceItem, EvidencePack
from app.services.search import tavily_search
from app.agents.llm import get_llm

RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce clean EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant and authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null.
- Keep snippets short (1–2 sentences max).
- Deduplicate by URL.
"""


def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def research_node(state: State) -> dict:
    llm = get_llm()
    queries = (state.get("queries") or [])[:10]

    raw: List[dict] = []
    for q in queries:
        raw.extend(tavily_search(q, max_results=6))

    if not raw:
        return {"evidence": []}

    extractor = llm.with_structured_output(EvidencePack)
    pack: EvidencePack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw results:\n{raw}"
                )
            ),
        ]
    )

    # Deduplicate by URL
    dedup: dict[str, EvidenceItem] = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    # For open_book, filter by recency
    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [
            e for e in evidence
            if (d := _iso_to_date(e.published_at)) and d >= cutoff
        ]

    return {"evidence": evidence}
