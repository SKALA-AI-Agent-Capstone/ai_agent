"""Compare the figures two readers saw on the same page (OCR and a vision model).

Only measured figures count (decimals and percents). A date such as "2024. 03. 19" or the end of a period "~ 04. 03"
is not a figure, so one reader spacing it differently is not a disagreement. The vision text is never used as the
page's text (a vision model repeats plausible numbers it did not see); the comparison only marks pages.
"""
from collections import Counter
import re

DATES = re.compile(r"\d{4}\s*[.\-/]\s*\d{1,2}\s*[.\-/]\s*\d{1,2}|(?<=[~\-–])\s*\d{1,2}\s*[.\-/]\s*\d{1,2}")


def numbers(text: str) -> list[str]:
    """Claim figures only: decimals and percents. A bare list index ('3 팔자 주름') is not a claim figure."""
    return re.findall(r"\d+\.\d+|\d+(?=\s*%)", (text or "").replace(",", ""))


def figures(text: str) -> list[str]:
    return numbers(DATES.sub(" ", text or ""))


def compare(vision: str, ocr: str) -> dict:
    """Figures one reader saw and the other did not. Equal sets agree, even if the words differ."""
    seen, read = Counter(figures(vision)), Counter(figures(ocr))
    vision_only = sorted((seen - read).elements())
    ocr_only = sorted((read - seen).elements())
    if vision_only or ocr_only:
        return {"status": "needs_review", "visionOnly": vision_only, "ocrOnly": ocr_only}
    return {"status": "match", "figures": sum(seen.values())}
