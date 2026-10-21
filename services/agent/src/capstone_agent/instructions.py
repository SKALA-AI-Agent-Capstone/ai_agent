"""System prompts (Harness: Instructions). They live in prompts/*.md so wording is versioned and reviewed
apart from code. A file holds one prompt; line breaks read as a single space, like a Markdown paragraph,
so a prompt can be wrapped one sentence per line without changing what the model receives."""
from functools import cache
from pathlib import Path

PROMPTS = Path(__file__).with_name("prompts")


@cache
def instruction(name: str) -> str:
    text = (PROMPTS / f"{name}.md").read_text(encoding="utf-8")
    return " ".join(line.strip() for line in text.splitlines() if line.strip())
