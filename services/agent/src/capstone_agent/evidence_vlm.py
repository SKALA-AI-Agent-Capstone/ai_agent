"""A second reader for scanned evidence pages: a vision model transcribes the page, and the figures are compared with
what OCR read. The vision text is never used as the page's text (a vision model repeats plausible numbers it did not
see); it only marks the pages where the two disagree, so a person checks those figures against the original.
"""
import asyncio

from pydantic import BaseModel

from capstone_shared.figure_check import compare, figures  # noqa: F401  (compare is part of this module's surface)

from .ad_reading import data_url
from .instructions import instruction

CALL_TIMEOUT_SECONDS = 70


class PageTranscript(BaseModel):
    lines: list[str]


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
