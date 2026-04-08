from __future__ import annotations
import re
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from app.models.state import State, GlobalImagePlan
from app.services.image import generate_image
from app.agents.llm import get_llm


# ──────────────────────────────────────────────
# Step 1: merge all sections into one markdown
# ──────────────────────────────────────────────

def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without a plan.")

    ordered_sections = [
        md for _, md in sorted(state["sections"], key=lambda x: x[0])
    ]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}

