"""A second reader for scanned evidence pages: a vision model transcribes the page, and the figures are compared with
what OCR read. The vision text is never used as the page's text (a vision model repeats plausible numbers it did not
see); it only marks the pages where the two disagree, so a person checks those figures against the original.
"""
import asyncio
from collections import Counter
import re

from pydantic import BaseModel

from .ad_reading import data_url, numbers
from .instructions import instruction

CALL_TIMEOUT_SECONDS = 70


class PageTranscript(BaseModel):
    lines: list[str]


# "2024. 03. 19", "2024-03-19" and the end of a period "~ 04. 03" are dates, not measured figures; one reader printing
# "04.03" and the other "04. 03" must not read as a disagreement.
DATES = re.compile(r"\d{4}\s*[.\-/]\s*\d{1,2}\s*[.\-/]\s*\d{1,2}|(?<=[~\-–])\s*\d{1,2}\s*[.\-/]\s*\d{1,2}")


def figures(text: str) -> list[str]:
    return numbers(DATES.sub(" ", text or ""))


def compare(vision: str, ocr: str) -> dict:
    """Figures (decimals and percents) one reader saw and the other did not. Equal sets agree, even if the words differ."""
    seen, read = Counter(figures(vision)), Counter(figures(ocr))
    vision_only = sorted((seen - read).elements())
    ocr_only = sorted((read - seen).elements())
    if vision_only or ocr_only:
        return {"status": "needs_review", "visionOnly": vision_only, "ocrOnly": ocr_only}
    return {"status": "match", "figures": sum(seen.values())}


def make_crosscheck(provider, loop):
    """A synchronous cross-check for the page reader, which runs in a worker thread: the model call runs on the Agent's
    event loop. Returns None when no model is configured, so pages are simply not cross-checked."""
    if provider is None or not getattr(provider, "client", None):
        return None

    def check(image, ocr_text: str) -> dict:
        future = asyncio.run_coroutine_threadsafe(
            provider.parse(instruction("evidence_transcribe"), PageTranscript, image=data_url(image),
                           text="이 쪽의 글자를 그대로 옮긴다."), loop)
        out = future.result(CALL_TIMEOUT_SECONDS)
        return compare(" ".join(out.lines), ocr_text)

    return check
