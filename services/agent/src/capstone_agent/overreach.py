"""Last look over the judged cards: a reviewer that only releases cards.

Earlier steps lean toward keeping a candidate (figure rows are forced through the judgment), so some cards ask
the reviewer to check things the ad never claims (scale ticks, step numbers, plain product description) or repeat
another card's fact. This call sees each card with its whole section and the section's other cards and releases
the ones a reviewer would drop, with a reason. It runs right after the judgment and before any basis is fetched,
so a released card costs no basis search, verification, severity or rewrite call. It never adds or rewrites cards.
"""
import asyncio

from pydantic import BaseModel

from .instructions import instruction

OVERREACH_BATCH = 12
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
        if v is not None and v.release:
            released.append({"id": f["id"], "phrase": f["phrase"], "textId": f.get("textId"), "reason": v.reason,
                             "source": "overreach_review"})
        else:
            kept.append(f)
    return kept, released, list(dict.fromkeys(warnings))
