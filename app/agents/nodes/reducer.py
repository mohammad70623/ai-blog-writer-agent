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


# ──────────────────────────────────────────────
# Step 2: decide where images should go
# ──────────────────────────────────────────────

DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images or diagrams would genuinely improve this blog post.

Rules:
- Maximum 3 images total.
- Only include an image if it materially aids understanding
  (e.g. architecture diagram, data flow, comparison table visual).
- Insert placeholders exactly as: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images are needed, return the markdown unchanged and images=[].
- Avoid purely decorative images.
- Write clear, descriptive image prompts (what to draw, style, labels).

Return strictly as GlobalImagePlan schema.
"""


def decide_images(state: State) -> dict:
    llm = get_llm()
    planner = llm.with_structured_output(GlobalImagePlan)

    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    image_plan: GlobalImagePlan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders where images would help, then propose prompts.\n\n"
                    f"{merged_md}"
                )
            ),
        ]
    )

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


# ──────────────────────────────────────────────
# Step 3: generate images and insert into markdown
# ──────────────────────────────────────────────

def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs") or []

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]

        try:
            img_path = generate_image(prompt=spec["prompt"], filename=filename)
            img_md = f"![{spec['alt']}]({img_path})\n*{spec['caption']}*"
        except Exception as e:
            # Graceful fallback: leave a descriptive block so the doc is still usable
            img_md = (
                f"> **[Image could not be generated]** {spec.get('caption', '')}\n"
                f">\n> **Alt:** {spec.get('alt', '')}\n"
                f">\n> **Prompt:** {spec.get('prompt', '')}\n"
                f">\n> **Error:** {e}\n"
            )

        md = md.replace(placeholder, img_md)

    filename = f"{_safe_slug(plan.blog_title)}.md"
    Path(filename).write_text(md, encoding="utf-8")

    return {"final": md}
