"""Last look over the judged cards: a reviewer that only releases cards.

Earlier steps lean toward keeping a candidate (figure rows are forced through the judgment), so some cards ask
the reviewer to check things the ad never claims (scale ticks, step numbers, plain product description) or repeat
another card's fact. This call sees each card with its whole section and the section's other cards and releases
the ones a reviewer would drop, with a reason. It runs right after the judgment and before any basis is fetched,
so a released card costs no basis search, verification, severity or rewrite call. It never adds or rewrites cards.
"""
import asyncio
import logging
import re

from pydantic import BaseModel

from .instructions import instruction

logger = logging.getLogger(__name__)

OVERREACH_BATCH = 12
# The reviewer may release a card the ad does not really claim, but never one that states what a reviewer must check.
# The prompt says so too, yet the model released "미백" and "4배 빠른 저분자!" in some runs, and a missed functional
# or numeric claim costs far more than one extra card. So this is a floor in code, whatever the model answers.
PROTECTED_TYPES = {"wrinkle", "whitening", "absolute", "patent", "drug"}
PROTECTED_TEXT = re.compile(
    r"미백|주름|자외선|탈모|여드름|튼살|염모|체모|아토피"                       # functional effects (심사·보고 대상)
    r"|\d+(?:[.,]\d+)?\s*(?:%|％|퍼센트|배)"                                  # figures stated as results or comparisons
    r"|\d+\s*(?:초|분|시간|일|주|개월|회)\s*(?:만에|만으로)"                         # how fast a result comes
    r"|\d+\s*(?:일|주|개월|회)\s*(?:후|간|동안).{0,12}(?:개선|효과|변화|감소|증가|완화)"  # a result after a period of use
    r"|최초|최고|유일|1위|No\.?\s*1|세계|업계|독보|완벽|영구|가장"               # absolute claims and comparisons
    r"|특허|인증|수상|심사|임상|시험|설문|검증|인체적용|피부과|의사|병원|약국",    # public basis and tests
    re.IGNORECASE)
KIND_NAMES = {"copy": "문구", "table": "목록 줄", "footnote": "각주", "certificate": "문서"}


class Release(BaseModel):
    id: str
    release: bool
    reason: str


class OverreachReview(BaseModel):
    verdicts: list[Release]


def section_listing(card: dict, texts: list[dict]) -> str:
    near = [t for t in texts if tile_of(t) == tile_of(card)][:14]
    return "\n".join(f"  - ({KIND_NAMES.get(t.get('kind'), '문구')}) {t['text'][:80]}" for t in near) or "  (없음)"


def tile_of(card: dict):
    return (card.get("location") or {}).get("tileId")


def group_by_section(findings: list[dict]) -> list[list[dict]]:
    groups = {}
    for f in findings:
        groups.setdefault(tile_of(f), []).append(f)
    return list(groups.values())


def protected(card: dict) -> bool:
    """A card the reviewer must not release: its type or its words say a reviewer has something to check."""
    from .vague_effect import effect_claim  # an effect on a skin concern ("모공 타이트닝", "노폐물 제거") is protected too
    phrase = card.get("phrase", "")
    return card.get("claimType") in PROTECTED_TYPES or bool(PROTECTED_TEXT.search(phrase)) or effect_claim(phrase)


async def review_overreach(provider, findings: list[dict], texts: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    """-> (kept, released, warnings). On a failed call every card of that batch is kept."""
    if not findings or not provider.client:
        return findings, [], []
    by_id = {t["id"]: t for t in texts}

    def listing(part):
        out = []
        for f in part:
            others = [g["phrase"][:60] for g in findings if g is not f and tile_of(g) == tile_of(f)]
            out.append(f"[{f['id']}] 카드 문구: {f['phrase']}\n줄 전체: {(by_id.get(f.get('textId')) or {}).get('text', f['phrase'])}"
                       f"\n카드 이유: {f.get('reason', '')}\n같은 구간:\n{section_listing(f, texts)}"
                       f"\n같은 구간의 다른 카드: {' / '.join(others) or '(없음)'}")
        return "\n\n".join(out)

    # A section's cards stay in one batch, so a repeated fact is seen next to the card it repeats.
    parts, part = [], []
    for cards in group_by_section(findings):
        if part and len(part) + len(cards) > OVERREACH_BATCH:
            parts.append(part)
            part = []
        part += cards
    if part:
        parts.append(part)
    outputs = await asyncio.gather(*(provider.parse(instruction("overreach_review"), OverreachReview, text=listing(p))
                                     for p in parts), return_exceptions=True)
    verdicts, warnings = {}, []
    for output in outputs:
        if isinstance(output, BaseException):
            warnings.append("OVERREACH_REVIEW_FAILED")
            continue
        verdicts.update({v.id: v for v in output.verdicts})
    kept, released = [], []
    for f in findings:
        v = verdicts.get(f["id"])
        if v is not None and v.release and protected(f):
            logger.warning("overreach release blocked for %s: %s", f["id"], f["phrase"][:40])
            warnings.append("OVERREACH_RELEASE_BLOCKED")
            kept.append(f)
        elif v is not None and v.release:
            released.append({"id": f["id"], "phrase": f["phrase"], "textId": f.get("textId"), "reason": v.reason,
                             "source": "overreach_review"})
        else:
            kept.append(f)
    return kept, released, list(dict.fromkeys(warnings))
