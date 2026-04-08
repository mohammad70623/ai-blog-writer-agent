from __future__ import annotations
import operator
from typing import TypedDict, List, Optional, Literal, Annotated
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Plan schemas
# ──────────────────────────────────────────────

class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence: what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]

