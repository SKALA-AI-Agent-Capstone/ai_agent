"""A plain ingredient claim ("울릉도 해양 심층수와 3종 히알루론산 듬뿍") is settled by the attached material, not by a test.

When a card only says the product contains certain ingredients, the reviewer's whole job is to see that the material
attached to the review says so too: another part of the ad (an ingredient list, a note) or any evidence document of the
product (an ingredient list, a test report's product description). If every named ingredient appears there, the card
leaves as settled with where it was found; a stated count ("3종") is reported when it could not be confirmed. If an
ingredient appears nowhere, or there is nothing to look at, the card stays with a check saying so. A card that also
claims an effect (a figure, a functional effect, a test, a comparison, an effect verb) is not a plain ingredient claim
and is left to the other checks.
"""
import re

from .checks import check, fallback_ingredients

# Effect words: the card claims more than presence, so the ingredient list alone cannot settle it.
EFFECT = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:%|％|퍼센트|배)|\d+\s*(?:초|분|시간|일|주|개월|회)\s*(?:만에|만으로|후|간|동안)"
    r"|미백|주름|자외선|탈모|여드름|튼살|염모|아토피|최초|최고|유일|1위|특허|인증|임상|시험|설문|검증|인체적용"
    r"|개선|완화|재생|치료|예방|강화|증가|감소|회복|제거|억제|효과|효능|리프팅|탄력|흡수율|침투")
COUNT = re.compile(r"(\d+)\s*종")
# Words that say the product contains something; "3종" counts too ("3종 히알루론산").
PRESENCE = re.compile(r"함유|듬뿍|듬뿐|가득|담은|담아|담겨|배합|넣은|넣어|들어|함량|추출|\d+\s*종")

# A name in ad copy and its spellings on an ingredient list (Korean INCI names differ from everyday names).
ALIASES = {
    "히알루론산": ["히알루론", "하이알루론", "하이알루로", "히알루로", "hyaluron"],
    "해양심층수": ["해양심층수", "심층수", "해수", "seawater", "sea water"],
    "콜라겐": ["콜라겐", "collagen"],
    "세라마이드": ["세라마이드", "ceramide"],
    "펩타이드": ["펩타이드", "펩티드", "peptide"],
    "나이아신아마이드": ["나이아신아마이드", "niacinamide"],
    "판테놀": ["판테놀", "panthenol"],
    "병풀": ["병풀", "센텔라", "centella"],
    "녹차": ["녹차", "차잎", "camellia sinensis"],
    "알로에": ["알로에", "aloe"],
    "스쿠알란": ["스쿠알란", "squalane"],
    "비타민c": ["아스코빅", "아스코빌", "ascorb"],
    "레티놀": ["레티놀", "레티닐", "retin"],
}


def squash(text: str) -> str:
    return re.sub(r"[\s·•\-_/]", "", text or "").lower()


def spellings(name: str) -> list[str]:
    """How a named ingredient may appear on a list: known aliases, else the name and its trailing words
    ("울릉도 해양 심층수" -> also "해양심층수", "심층수")."""
    key = squash(name)
    for canonical, alias in ALIASES.items():
        if canonical in key or any(a in key for a in alias):
            return alias
    words = [w for w in re.split(r"\s+", name.strip()) if w]
    tails = [squash("".join(words[i:])) for i in range(len(words))]
    return [t for t in dict.fromkeys([key] + tails) if len(t) >= 2]


def attachments(card: dict, texts: list[dict], documents: list[dict]) -> list[tuple[str, str]]:
    """(where, text) of every attached material except the card's own line: the rest of the ad and each evidence document.
    The card's own line names the ingredient by definition, so it cannot confirm itself."""
    ad = [t.get("text", "") for t in texts if t.get("id") != card.get("textId") and t.get("text", "") != card.get("phrase")]
    found = [("광고", "\n".join(ad))] if any(ad) else []
    return found + [(doc.get("name") or "증빙", doc.get("text", "")) for doc in documents or [] if doc.get("text")]


def mentions(text: str, alias: list[str]) -> list[str]:
    """Items of the text (comma or line separated) that carry the ingredient, e.g. each hyaluronic acid on a list."""
    return list(dict.fromkeys(squash(i) for i in re.split(r"[,，、\n]", text) if any(a in squash(i) for a in alias)))


def names_ingredient(phrase: str) -> bool:
    key = squash(phrase)
    return any(a in key for alias in ALIASES.values() for a in alias) or bool(fallback_ingredients(phrase))


def ingredient_claim(phrase: str) -> bool:
    """Copy that says the product contains a named ingredient and claims nothing more."""
    return bool(phrase) and not EFFECT.search(phrase) and bool(PRESENCE.search(phrase)) and names_ingredient(phrase)


def rescue_ingredient_claims(findings: list[dict], dismissed: list[dict], texts: list[dict]) -> tuple[list[dict], list[dict]]:
    """-> (findings, dismissed). An earlier step may drop a plain ingredient claim as harmless copy, but whether it is
    harmless depends on the attached material, so it goes on as an ingredient card and settle_ingredient_claims decides."""
    by_id = {t["id"]: t for t in texts}
    taken = {f.get("textId") for f in findings}
    rescued, left = [], []
    for item in dismissed:
        block = by_id.get(item.get("textId"))
        if block is None or item.get("textId") in taken or not ingredient_claim(item.get("phrase", "")):
            left.append(item)
            continue
        taken.add(item["textId"])
        phrase = item["phrase"]
        start = max(0, block["text"].find(phrase))
        rescued.append({"id": f"ing-{block['id']}", "phrase": phrase, "start": start, "end": start + len(phrase),
                        "location": block.get("location"), "textId": block["id"], "severity": "low", "baseSeverity": "low",
                        "claimType": "ingredient", "requiredChecks": ["ingredient"],
                        "reason": "제품에 들어 있다고 말하는 성분 표현입니다. 첨부 자료에 그 성분이 있는지 확인합니다.",
                        "title": "성분 함유 표현", "basis": [], "baseBasis": [], "confidence": None, "rewrites": [],
                        "references": [], "category": "확인할 광고 주장", "legalStatus": "not_evaluated", "escalation": None,
                        "source": "ingredient_claim"})
    return findings + rescued, left


def plain_ingredient_claim(card: dict) -> bool:
    return card.get("claimType") == "ingredient" and not EFFECT.search(card.get("phrase", ""))


def names_of(card: dict) -> list[str]:
    """Ingredients the card names: what the safety step extracted, else a word-shape guess, plus every known name the
    phrase spells out (the extraction sometimes returns only one of two ingredients)."""
    phrase = card.get("phrase", "")
    named = [c["detail"]["ingredient"] for c in card.get("checks", []) if c.get("type") == "ingredient" and c.get("detail", {}).get("ingredient")]
    named = named or fallback_ingredients(phrase)
    known = [canonical for canonical, alias in ALIASES.items() if any(a in squash(phrase) for a in alias)
             and not any(any(a in squash(n) for a in alias) for n in named)]
    return list(dict.fromkeys(named + known))


def presence(card: dict, sources: list[tuple[str, str]]) -> tuple[str, str, dict]:
    """-> (status, summary, detail). status: in_attachments | not_in_attachments | needs_input."""
    names = names_of(card)
    if not names:
        return "needs_input", "문구에서 성분 이름을 특정하지 못해 첨부 자료와 대조하지 못했습니다.", {}
    if not sources:
        return "needs_input", "광고의 다른 부분이나 제품 증빙이 없어 성분이 들어 있는지 대조하지 못했습니다.", {"ingredients": names}
    found, missing = {}, []
    for name in names:
        alias = spellings(name)
        hits = [(where, items) for where, text in sources if (items := mentions(text, alias))]
        if hits:
            found[name] = {"where": list(dict.fromkeys(w for w, _ in hits)), "items": [i for _, items in hits for i in items][:8]}
        else:
            missing.append(name)
    detail = {"ingredients": names, "found": found, "missing": missing}
    if missing:
        return "not_in_attachments", f"광고의 다른 부분과 제품 증빙 어디에도 {', '.join(missing)}이(가) 들어 있다는 내용이 없습니다.", detail
    places = ", ".join(dict.fromkeys(w for f in found.values() for w in f["where"]))
    summary = f"{', '.join(names)}이(가) 들어 있다는 내용이 첨부 자료({places})에 있습니다."
    wanted = COUNT.search(card.get("phrase", ""))
    if wanted and not any(len(f["items"]) >= int(wanted.group(1)) for f in found.values()):
        summary += f" 다만 {wanted.group(1)}종인지는 자료에서 확인하지 못했습니다."
    return "in_attachments", summary, detail


def settle_ingredient_claims(findings: list[dict], texts: list[dict], documents: list[dict]) -> tuple[list[dict], list[dict]]:
    """-> (kept, settled). A plain ingredient claim whose ingredients the attached material confirms leaves as settled
    (no problem); otherwise it stays with an ingredient_list check. Other cards are untouched."""
    kept, settled = [], []
    for f in findings:
        if not plain_ingredient_claim(f):
            kept.append(f)
            continue
        status, summary, detail = presence(f, attachments(f, texts, documents))
        result = check("ingredient_list", status, summary, source="첨부 자료(광고, 제품 증빙)", detail=detail)
        others = [c for c in f.get("checks", []) if c.get("type") not in ("substantiation", "ingredient_list")]
        # Only a banned or restricted ingredient, or another check that needs work, keeps a confirmed card. The safety
        # step failing to name the ingredient ("needs_input") is not a reason: this check found them by name.
        blocking = [c for c in others if c.get("attention") and not (c.get("type") == "ingredient" and c.get("status") in ("needs_input", "unavailable"))]
        if status == "in_attachments" and not blocking:
            settled.append({"id": f["id"], "phrase": f["phrase"], "textId": f.get("textId"), "source": "ingredient_listed",
                            "reason": summary})
        else:
            # A presence claim is not a test result, so the test-oriented substantiation check is replaced by this one.
            kept.append({**f, "checks": others + [result]})
    return kept, settled
