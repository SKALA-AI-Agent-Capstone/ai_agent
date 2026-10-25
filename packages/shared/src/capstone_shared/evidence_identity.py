"""Who issued an evidence report, under which code and version, and which products it covers on which pages.

The issuer comes first: a clinical research centre prints its codes in its own format, so a known issuer's
format picks the code out of the text exactly (no digits glued on from the next line, no page numbers). A code
is trusted when it shows up the same on several pages. An issuer nobody has registered falls back to labelled
codes found by a vote across pages, and the result says it still needs a person to confirm.

Codes may differ from issuer to issuer; nothing here unifies them. The registry below is the list of formats
seen so far, one line per issuer.
"""
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from difflib import SequenceMatcher
import math
import re

from .reading import squash


@dataclass(frozen=True)
class Issuer:
    key: str
    name: str
    mention: re.Pattern
    codes: tuple[tuple[str, re.Pattern], ...]  # (label, pattern run on whitespace-free text)


ISSUERS = (
    Issuer("ksrc", "KSRC 한국피부임상연구센터", re.compile(r"한국피부임상연구센터|KSRC"),
           (("연구 코드", re.compile(r"HBSE-[A-Z]+-\d{5}")),)),
    Issuer("probem", "프로브엠 피부임상연구센터", re.compile(r"프로브엠"),
           # The two letters name the test (SA 자극, LI 리프팅 ...); a study of any test keeps its number.
           (("연구번호", re.compile(r"PM-\d{2}J\d{2}-[A-Z]{2}\d{3}")),
            ("시료번호", re.compile(r"PM-\d{2}SA-\d{3}")))),
)

LABELLED_CODE = re.compile(
    r"(?P<label>연\s*구\s*코\s*드|연\s*구\s*번\s*호|시\s*험\s*번\s*호|보\s*고\s*서\s*번\s*호|관\s*리\s*번\s*호"
    r"|Report\s*No\.?|Study\s*(?:No\.?|ID|Code))\s*[:：]?\s*(?P<code>[A-Za-z0-9][A-Za-z0-9\-_/]{4,})")
REVISION_DATE = re.compile(r"(20\d{2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})")
VERSION_NO = re.compile(r"Version\s*No\.?\s*:?\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
COVER_MONTH = re.compile(r"(20\d{2})\s*년\s*(\d{1,2})\s*월")
PRODUCT_LIST = re.compile(r"^\s*#\s*(\d+)\s*[:：]\s*(\S.*?)\s*$", re.MULTILINE)
# Korean reports often open a quote with ’ as well as ‘.
QUOTED = re.compile(r"[“‘’”\"']([^”’\"']{4,60})[”’\"']")
RESULT = re.compile(r"[“‘’”\"']([^”’\"']{4,80})[”’\"']\s*(?:가|이|은|는)?\s*[^.%]{0,30}?(\d+(?:\.\d+)?)\s*%\s*(?:의\s*)?변화율")
# "2. 즉각적인 피부 투명도(맑기) 평가": a numbered heading, not the tail of "p<0.05). 2." before it.
ITEM_HEADING = re.compile(r"(?<![\d.])\d+(?:-\d+)?\.\s+([가-힣A-Za-z](?:[^.]|(?<=\d)\.(?=\d)){1,38}?평가)")
MATCH_MIN = 0.85
FUZZY_WINDOW = 0.88


@dataclass
class Identity:
    issuer_key: str | None
    issuer_name: str | None
    codes: list[dict] = field(default_factory=list)
    version: str | None = None
    issued_on: str | None = None


def detect_issuer(pages: list[str]) -> Issuer | None:
    if not pages:
        return None
    needed = min(len(pages), max(2, math.ceil(len(pages) * 0.1)))
    counts = {issuer: sum(bool(issuer.mention.search(squash(page))) for page in pages) for issuer in ISSUERS}
    best = max(counts, key=counts.get)
    return best if counts[best] >= needed else None


def _vote(per_page: list[set[str]], single_page_ok: bool) -> tuple[str, int, str, list[str]] | None:
    counts = Counter(value for values in per_page for value in values)
    if not counts:
        return None
    ranked = counts.most_common()
    top = ranked[0][1]
    leaders = sorted(value for value, n in ranked if n == top)
    value, pages = leaders[0], top
    if len(leaders) > 1:
        status = "ambiguous"
    elif pages >= 2 or single_page_ok:
        status = "confirmed"
    else:
        status = "single_page"
    return value, pages, status, leaders[1:]


def extract_codes(pages: list[str], issuer: Issuer | None) -> list[dict]:
    codes = []
    if issuer is not None:
        squashed = [squash(page) for page in pages]
        for label, pattern in issuer.codes:
            voted = _vote([set(pattern.findall(text)) for text in squashed], single_page_ok=len(pages) == 1)
            if voted:
                value, count, status, others = voted
                codes.append({"label": label, "value": value, "pages": count, "status": status,
                              **({"alternatives": others} if others else {})})
        return codes
    # An issuer nobody registered: labelled codes only, never trusted without a person's confirmation.
    found: dict[str, list[set[str]]] = {}
    for page in pages:
        per_page: dict[str, set[str]] = {}
        for line in page.splitlines():
            for match in LABELLED_CODE.finditer(line):
                code = match.group("code").rstrip("-_/")
                if len(code) >= 6 and re.search(r"\d", code):
                    per_page.setdefault(re.sub(r"\s+", "", match.group("label")), set()).add(code)
        for label, values in per_page.items():
            found.setdefault(label, []).append(values)
    for label, per_page in found.items():
        value, count, status, others = _vote(per_page, single_page_ok=False)
        codes.append({"label": label, "value": value, "pages": count,
                      "status": "probable" if count >= 3 and status == "confirmed" else "unconfirmed",
                      **({"alternatives": others} if others else {})})
    return codes


def extract_version(pages: list[str]) -> tuple[str | None, str | None]:
    """(version label, issue date). A revision table gives the latest approval date; a cover gives the month."""
    parts, issued = [], None
    head = "\n".join(pages[:3])
    if match := VERSION_NO.search(head):
        parts.append(f"Version {match.group(1)}")
    for page in pages:
        if "개정이력" in squash(page):
            dates = []
            for year, month, day in REVISION_DATE.findall(page):
                try:
                    dates.append(date(int(year), int(month), int(day)))
                except ValueError:
                    continue
            if dates:
                parts.append(f"개정 이력 {len(dates)}건")
                issued = max(dates).isoformat()
                break
    if issued is None and pages and (cover := COVER_MONTH.search(pages[0])):
        issued = f"{cover.group(1)}-{int(cover.group(2)):02d}"
    if issued:
        parts.append(issued)
    return (" · ".join(parts) or None), issued


def identify(pages: list[str]) -> Identity:
    issuer = detect_issuer(pages)
    version, issued = extract_version(pages)
    return Identity(issuer.key if issuer else None, issuer.name if issuer else None,
                    extract_codes(pages, issuer), version, issued)


def group_ranges(numbers: list[int]) -> list[list[int]]:
    ranges: list[list[int]] = []
    for number in sorted(set(numbers)):
        if ranges and number == ranges[-1][1] + 1:
            ranges[-1][1] = number
        else:
            ranges.append([number, number])
    return ranges


def product_names(pages: list[str]) -> list[str]:
    """Test products listed as '#1: name' near the front, else the first quoted name on the cover."""
    head = "\n".join(pages[:12])
    listed: dict[int, str] = {}
    for number, name in PRODUCT_LIST.findall(head):
        listed.setdefault(int(number), re.sub(r"\s+", " ", name).strip())
    if listed:
        return [listed[key] for key in sorted(listed)]
    if pages:
        if match := QUOTED.search(re.sub(r"\s+", " ", pages[0])):
            if re.search(r"[가-힣]", match.group(1)):
                return [match.group(1).strip()]
    return []


def _distinctive(names: list[str]) -> list[list[str]]:
    """Per product, the words no other product's name has (핑크, 리프팅 ...): a page naming two of them is about it."""
    tokens = [{t for t in re.split(r"\s+", name) if len(t) >= 2} for name in names]
    return [sorted(own - set().union(*(tokens[j] for j in range(len(names)) if j != i))) for i, own in enumerate(tokens)]


def _fuzzy_contains(page: str, name: str) -> bool:
    if name in page:
        return True
    size = len(name)
    if size < 6 or len(page) < size:
        return False
    anchor = name[:2]
    for start in (m.start() for m in re.finditer(re.escape(anchor), page)):
        window = page[start:start + size]
        if SequenceMatcher(None, window, name, autojunk=False).ratio() >= FUZZY_WINDOW:
            return True
    return False


def product_pages(pages: list[str], names: list[str]) -> list[list[int]]:
    squashed = [squash(page) for page in pages]
    own_words = _distinctive(names) if len(names) > 1 else [[] for _ in names]
    result = []
    for name, words in zip(names, own_words):
        norm, found = squash(name), []
        for number, text in enumerate(squashed, 1):
            by_name = _fuzzy_contains(text, norm)
            by_words = len(words) >= 2 and sum(squash(w) in text for w in words) >= 2
            if by_name or by_words:
                found.append(number)
        result.append(found)
    return result


def match_product(registered: str, names: list[str]) -> tuple[int, float] | None:
    """Which product of the report is the registered one. Spacing and a few misspelt letters do not matter."""
    target = squash(registered)
    if not target:
        return None
    words = [squash(w) for w in re.split(r"\s+", registered.strip()) if len(w) >= 2]
    best: tuple[int, float] | None = None
    for index, name in enumerate(names):
        candidate = squash(name)
        score = SequenceMatcher(None, target, candidate, autojunk=False).ratio()
        if words and all(word in candidate for word in words):
            score = max(score, 0.9)
        if best is None or score > best[1]:
            best = (index, score)
    return best if best and best[1] >= MATCH_MIN else None


def extract_results(pages: list[str], names: list[str]) -> list[dict]:
    """'<product> ... 9.71% 변화율' sentences, with the measured item named by the heading before them."""
    if not names:
        return []
    full, offsets = "", []
    for text in pages:
        offsets.append(len(full))
        full += re.sub(r"\s+", " ", text) + " "
    results = []
    for match in RESULT.finditer(full):
        found = match_product(match.group(1), names)
        if found is None:
            continue
        headings = ITEM_HEADING.findall(full[max(0, match.start() - 500):match.start()])
        after = full[match.end():match.end() + 60]
        results.append({"product": names[found[0]], "item": headings[-1].strip() if headings else None,
                        "changePct": float(match.group(2)), "page": bisect_right(offsets, match.start()),
                        "significant": "유의" in after})
    return results
