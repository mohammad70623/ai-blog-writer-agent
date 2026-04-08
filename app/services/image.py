from __future__ import annotations
import urllib.parse
from pathlib import Path
import requests
from app.core.config import IMAGES_DIR


def generate_image(prompt: str, filename: str) -> Path:
    """
    Generate an image using Pollinations.AI (free, no API key required).
    Saves to data/images/<filename> and returns the path.

    Pollinations URL format:
        https://image.pollinations.ai/prompt/<encoded_prompt>?width=1024&height=768&nologo=true
    """
    images_dir = Path(IMAGES_DIR)
    images_dir.mkdir(parents=True, exist_ok=True)
    out_path = images_dir / filename

    if out_path.exists():
        return out_path

    encoded_prompt = urllib.parse.quote(prompt)
    url = (
        f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        f"?width=1024&height=768&nologo=true&model=flux"
    )

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    out_path.write_bytes(response.content)
    return out_path
