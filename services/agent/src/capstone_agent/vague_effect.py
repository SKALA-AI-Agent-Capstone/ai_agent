"""Generic care copy that does not say which effect it means ("피부 결을 개선시켜", "영양이 빈틈없이 흡수되어") leaves the
review: there is nothing specific for a reviewer to check or fix. A card stays when it names something checkable: a
figure, a functional effect (or its usual names such as 브라이트닝), a medical or drug-like effect, an absolute
claim or comparison, a test, a patent or certification, or an ingredient (settled separately by ingredient_presence).
"""
import re

from .checks import claimed_functions
from .ingredient_presence import ingredient_claim, names_ingredient
from .overreach import PROTECTED_TEXT

# No type stays on the judge's word alone: the words of the card must back it. "광채 집중 케어" judged as whitening,
# "활력인자 29종에 도움" judged as an ingredient claim, or "외부의 유해 자극으로부터 피부를 보호" judged as drug-like
# name no functional effect, no ingredient and no medical effect; protecting, moisturising or cleansing the skin is what
# every cosmetic does.
PATENT = re.compile(r"특허|특허청|출원|실용신안|등록\s*번호|KIPRIS", re.IGNORECASE)
# An absolute claim needs an absolute word; "영원히 빛나는" is a figure of speech, not a ranking or a guarantee.
ABSOLUTE = re.compile(r"최초|최고|유일|1위|No\.?\s*1|세계|업계|독보|완벽|영구|가장|100\s*%"
                      r"|(?:평생|영원히|영원한)\s*.{0,8}(?:지속|유지|효과|보장)", re.IGNORECASE)  # a lasting effect is a claim
# "리프팅" alone is mood ("탄력 넘치는 리프팅 시너지 조합"); "리프팅 효과", "리프팅돼" says the skin is lifted.
# "즉시"/"즉각" alone is pace ("즉각적으로 수분을 공급"); with a change it promises ("사용 즉시 피부가 젊어집니다") it is specific.
# A skin concern alone ("피지 분비가 많은 T존", "각질을 부드럽게 정돈") describes the skin; it is an effect claim when an
# effect verb acts on it ("모공 수축", "노폐물 제거", "피지 조절").
CONCERN_EFFECT = r"(?:모공|피지|각질|노폐물|블랙헤드|트러블)[^,.!]{0,8}?(?:제거|조절|흡착|수축|타이트닝|감소|억제|축소|딥\s*클렌징)"
SPECIFIC = re.compile(
    r"브라이트닝|화이트닝|기미|잡티|주근깨|노화|안티\s*에이징|항산화|리프팅\s*(?:효과|개선|되|돼|해)|장벽|" + CONCERN_EFFECT +
    r"|\d+\s*(?:년|살|세)|젊어|어려지|회춘|(?:즉시|즉각|바로)[^,.!]{0,12}?(?:효과|개선|변화|젊어|사라|완화|감소)"
    r"|재생|세포|치료|완치|예방|질환|질병|염증|상처|흉터|피부염|줄기|DNA|유전자|호르몬|혈류|혈액|면역|항균|살균|소염|진통|의약|처방"
    r"|필러|보톡스|시술|주사",                                                         # medical procedures used as comparisons
    re.IGNORECASE)


def vague(card: dict) -> bool:
    phrase = card.get("phrase", "")
    if PROTECTED_TEXT.search(phrase) or SPECIFIC.search(phrase):
        return False
    if card.get("claimType") == "patent" and PATENT.search(phrase):
        return False
    if card.get("claimType") in ("wrinkle", "whitening") and claimed_functions(phrase):
        return False
    if card.get("claimType") == "absolute" and ABSOLUTE.search(phrase):
        return False
    return not (card.get("claimType") == "ingredient" and names_ingredient(phrase))


# A paragraph this long is an explanation or a seller's notice, not a claim a reviewer can mark and fix. It stays only
# when it states a figure, a functional effect or a medical effect itself.
NOTICE_LENGTH = 200
EFFECT_ITSELF = re.compile(r"\d+(?:[.,]\d+)?\s*(?:%|％|배)|" + CONCERN_EFFECT +
                           r"|미백|주름|자외선|탈모|여드름|재생|치료|세포|브라이트닝|화이트닝|항산화|안티\s*에이징|필러|보톡스|시술|주사")


def notice(card: dict) -> bool:
    phrase = card.get("phrase", "")
    return len(phrase) > NOTICE_LENGTH and not EFFECT_ITSELF.search(phrase)


def effect_claim(phrase: str) -> bool:
    """Copy that itself states an effect a reviewer checks: a figure, a functional or medical effect, an effect on a
    skin concern. Used to bring back such a line when the judge dropped it."""
    return bool(phrase) and len(phrase) <= NOTICE_LENGTH and bool(EFFECT_ITSELF.search(phrase))


# The line the card was taken from cites a test ("시험 결과 : 피부 일차 자극 평가에 대한 인체적용시험 결과, 저자극 제품군에
# 속하는 것으로 판단"). The extraction may hand over only the tail clause ("저자극 제품군에 속하는 것으로 판단"), which says
# nothing specific alone; the line as a whole is a test result a submitted document can confirm or not.
TEST_CITED = re.compile(r"시험\s*(?:결과|성적)|인체\s*적용\s*시험|임상|연구\s*결과|시험\s*기관")


def source_line(card: dict, texts: list[dict] | None) -> str:
    block = next((t for t in texts or [] if t.get("id") == card.get("textId")), None)
    return (block or {}).get("text", "") or card.get("phrase", "")


def cites_test(card: dict, texts: list[dict] | None) -> bool:
    line = source_line(card, texts)
    return len(line) <= NOTICE_LENGTH and bool(TEST_CITED.search(line))


def needs_evidence(card: dict) -> dict:
    checks = card.get("requiredChecks", [])
    return card if "substantiation" in checks else {**card, "requiredChecks": [*checks, "substantiation"]}


def dismiss_vague_effects(findings: list[dict], texts: list[dict] | None = None) -> tuple[list[dict], list[dict]]:
    """-> (kept, dismissed). The dismissed entries say why, like every other dismissal. A card whose line cites a test
    result is kept and goes to the evidence check, whatever its own words say."""
    kept, dismissed = [], []
    for f in findings:
        if cites_test(f, texts):
            kept.append(needs_evidence(f))
        elif notice(f):
            dismissed.append({"id": f["id"], "phrase": f["phrase"], "textId": f.get("textId"), "source": "notice_text",
                              "reason": "판매처 안내나 설명문처럼 긴 글이라 광고 주장으로 보지 않았습니다."})
        elif vague(f):
            dismissed.append({"id": f["id"], "phrase": f["phrase"], "textId": f.get("textId"), "source": "vague_effect",
                              "reason": "어떤 효능인지 구체적으로 말하지 않는 일반적인 관리 표현이라 확인할 사실이 없습니다."})
        else:
            kept.append(f)
    return kept, dismissed


def specific(card: dict) -> bool:
    """The card names something checkable, whatever its type: a figure, a functional or drug-like effect, a comparison,
    a test or patent, or a named ingredient."""
    phrase = card.get("phrase", "")
    return bool(PROTECTED_TEXT.search(phrase) or SPECIFIC.search(phrase) or claimed_functions(phrase)
                or (card.get("claimType") == "ingredient" and names_ingredient(phrase)) or ingredient_claim(phrase))


def set_aside_low_risk(findings: list[dict], texts: list[dict] | None = None) -> tuple[list[dict], list[dict]]:
    """-> (sent on, set aside). A card the first pass rated low risk is not judged further, unless it names something
    checkable: a first-pass "low" on "볼륨 개선도 92.7%" or "미백" would otherwise drop a real claim."""
    sent, aside = [], []
    for f in findings:
        if f.get("parseRisk") == "low" and not specific(f) and not cites_test(f, texts):
            aside.append({"id": f["id"], "phrase": f["phrase"], "textId": f.get("textId"), "source": "low_at_parse",
                          "reason": "1차 판단에서 위험이 낮다고 본 일반 표현이라 다음 단계로 보내지 않았습니다."})
        else:
            sent.append(f)
    return sent, aside


# A footnote naming a human test ("*투집 포어본드 일시적 모공(면적/볼륨/깊이) 개선 인체적용시험 *연구 기간 ... | 30~50대 여성")
# says that test was done on these terms. When no claim of its section goes to the evidence check, the footnote does.
HUMAN_TEST = re.compile(r"인체\s*적용\s*시험|임상\s*시험")


def test_footnote_cards(findings: list[dict], texts: list[dict]) -> list[dict]:
    from .evidence import where
    checked = {where(f) for f in findings if "substantiation" in f.get("requiredChecks", [])}
    taken = {f.get("textId") for f in findings}
    cards = []
    for block in texts:
        line = block.get("text", "").strip()
        if (block.get("role") != "condition" or block["id"] in taken or where(block) in checked
                or len(line) > 2 * NOTICE_LENGTH or not HUMAN_TEST.search(line)):
            continue
        checked.add(where(block))
        cards.append({"id": f"note-{block['id']}", "phrase": line, "start": 0, "end": len(line),
                      "location": block.get("location"), "textId": block["id"], "severity": "low", "baseSeverity": "low",
                      "claimType": "clinical", "requiredChecks": ["substantiation"],
                      "reason": "광고 각주가 이 조건의 인체적용시험을 했다고 적고 있어, 제출한 시험 자료와 맞는지 확인해야 합니다.",
                      "title": "광고에 적힌 시험", "basis": [], "baseBasis": [], "confidence": None, "rewrites": [],
                      "references": [], "category": "확인할 광고 주장", "legalStatus": "not_evaluated", "escalation": None,
                      "source": "test_footnote"})
    return cards
