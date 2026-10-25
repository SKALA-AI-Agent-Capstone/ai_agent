"""A card that only says what the product contains is settled when the attached material (another part of the ad, or a
product evidence document) says the product contains it. The card's own line cannot confirm itself."""
from types import SimpleNamespace

import capstone_agent.review_engine as engine
from capstone_agent.ingredient_presence import plain_ingredient_claim, settle_ingredient_claims, spellings

PHRASE = "울릉도 해양 심층수와 3종 히알루론산 듬뿍"


def card(phrase=PHRASE, claim_type="ingredient", checks=None, text_id="t1"):
    return {"id": "f1", "phrase": phrase, "claimType": claim_type, "textId": text_id,
            "checks": checks if checks is not None else [
                {"type": "ingredient", "status": "not_listed", "attention": False, "detail": {"ingredient": "해양 심층수"}},
                {"type": "ingredient", "status": "not_listed", "attention": False, "detail": {"ingredient": "히알루론산"}},
                {"type": "substantiation", "status": "evidence_not_covering", "attention": True, "detail": {}}]}


AD_LINE = {"id": "t1", "text": PHRASE + "한 후에도 피부에 촉촉함을 남겨주죠"}
AD_LIST = {"id": "t9", "text": "전성분: 정제수, 해수, 글리세린, 소듐하이알루로네이트, 하이알루로닉애씨드, 하이드롤라이즈드하이알루로닉애씨드"}


def test_names_on_lists_differ_from_names_in_copy():
    assert "하이알루로" in spellings("히알루론산") and "해수" in spellings("울릉도 해양 심층수")
    assert "산양유" in spellings("산양유")                         # an unknown name is looked up as written


def test_a_plain_ingredient_claim_confirmed_elsewhere_in_the_ad_is_settled():
    kept, settled = settle_ingredient_claims([card()], [AD_LINE, AD_LIST], [])
    assert kept == [] and settled[0]["source"] == "ingredient_listed"
    assert "광고" in settled[0]["reason"] and "3종인지는" not in settled[0]["reason"]   # three forms are on the list


def test_a_product_evidence_document_that_says_so_settles_it_too():
    doc = {"name": "디퍼 시험성적서", "text": "시험 제품 전성분 정제수, 글리세린, 해양심층수, 소듐하이알루로네이트"}
    kept, settled = settle_ingredient_claims([card()], [AD_LINE], [doc])
    assert kept == [] and "디퍼 시험성적서" in settled[0]["reason"]
    assert "3종인지는 자료에서 확인하지 못했습니다" in settled[0]["reason"]      # only one form found: said, not blocked


def test_the_cards_own_line_cannot_confirm_itself_and_the_card_stays_with_a_reason():
    kept, settled = settle_ingredient_claims([card()], [AD_LINE], [])
    assert settled == [] and len(kept) == 1
    types = {c["type"]: c["status"] for c in kept[0]["checks"]}
    assert types == {"ingredient": "not_listed", "ingredient_list": "needs_input"}      # test-oriented check replaced
    kept, _ = settle_ingredient_claims([card()], [AD_LINE, {"id": "t2", "text": "글리세린, 정제수"}], [])
    assert {c["type"]: c["status"] for c in kept[0]["checks"]}["ingredient_list"] == "not_in_attachments"
    assert next(c for c in kept[0]["checks"] if c["type"] == "ingredient_list")["attention"] is True


def test_a_claim_with_an_effect_or_a_banned_ingredient_is_not_settled_this_way():
    assert not plain_ingredient_claim(card("히알루론산으로 수분 30% 개선"))
    assert not plain_ingredient_claim(card("히알루론산이 주름을 펴 줍니다"))
    assert not plain_ingredient_claim(card(PHRASE, claim_type="exaggeration"))
    effect = card("히알루론산으로 수분 30% 개선")
    assert settle_ingredient_claims([effect], [AD_LINE, AD_LIST], []) == ([effect], [])
    banned = card(checks=[{"type": "ingredient", "status": "prohibited", "attention": True, "detail": {"ingredient": "히알루론산"}}])
    kept, settled = settle_ingredient_claims([banned], [AD_LINE, AD_LIST], [])
    assert settled == [] and kept[0]["checks"][0]["status"] == "prohibited"


def test_without_named_ingredients_or_material_the_check_says_what_is_missing():
    kept, _ = settle_ingredient_claims([card("좋은 성분 듬뿍", checks=[])], [AD_LINE, AD_LIST], [])
    assert kept[0]["checks"][-1]["status"] == "needs_input"


async def test_an_ingredient_card_fetches_the_products_evidence_even_if_the_model_skips_it(monkeypatch):
    calls = []

    async def mcp(name, arguments, request=None):
        calls.append(name)
        return {"status": "extracted", "documents": []}

    class Provider:
        client = True

        async def call_with_tools(self, messages, tools):
            return SimpleNamespace(content="성분 소개라 증빙 대조가 필요 없음", tool_calls=None)

    monkeypatch.setattr(engine, "mcp_call", mcp)
    state = {"tenantId": "t", "reviewId": "r", "product": {"name": "크림"}, "texts": [AD_LINE],
             "findings": [{"phrase": PHRASE, "reason": "성분", "claimType": "ingredient", "requiredChecks": ["ingredient"]}]}
    request = SimpleNamespace(assets=[], productEvidenceCount=1)
    *_, trace = await engine.select_business_tools(Provider(), state, request)
    assert calls == ["structure_evidence"] and {"tool": "structure_evidence", "called": True, "required": True} in trace


def test_only_copy_that_says_a_named_ingredient_is_in_the_product_counts_as_an_ingredient_claim():
    from capstone_agent.ingredient_presence import ingredient_claim
    assert ingredient_claim("울릉도 해양 심층수와 3종 히알루론산 듬뿐 세안 후에도")    # OCR misread of 듬뿍 still counts
    assert ingredient_claim("히알루론산 함유") and ingredient_claim("콜라겐 가득 담은 크림")
    assert ingredient_claim("모란 추출물 함유") and ingredient_claim("셀 소스™ 성분을 함유한 에센스")
    assert not ingredient_claim("모란 추출물") and not ingredient_claim("다마스크 로즈 워터")   # a bare name claims nothing
    for other in ["시간이 경과해도 마르지 않는 촉촉한 크림제형", "식물소재만 담아 자극걱정없이 순하게 진정 케어",
                  "브레아트 애플사이더 100%", "히알루론산으로 주름 개선", "수분 및 영양 공급"]:
        assert not ingredient_claim(other), other


def test_a_dropped_ingredient_claim_goes_on_as_a_card_for_the_attachment_check():
    from capstone_agent.ingredient_presence import rescue_ingredient_claims
    texts = [{"id": "t1", "text": PHRASE + " 세안 후에도", "location": {"tileId": "a:p1:t1"}}, AD_LIST]
    dismissed = [{"id": "f9", "phrase": PHRASE, "textId": "t1", "source": "model", "reason": "성분 소개"},
                 {"id": "f8", "phrase": "촉촉한 크림제형", "textId": "t1", "source": "model", "reason": "사용감"}]
    findings, left = rescue_ingredient_claims([], dismissed, texts)
    assert [f["phrase"] for f in findings] == [PHRASE] and findings[0]["claimType"] == "ingredient"
    assert findings[0]["requiredChecks"] == ["ingredient"] and findings[0]["location"] == {"tileId": "a:p1:t1"}
    assert [d["id"] for d in left] == ["f8"]                       # the other dismissal stays dismissed
    kept, settled = settle_ingredient_claims(findings, texts, [])
    assert kept == [] and settled[0]["source"] == "ingredient_listed"   # the ad's ingredient list settles it
    findings, _ = rescue_ingredient_claims([{"id": "x", "textId": "t1", "phrase": "다른 카드"}], dismissed, texts)
    assert len(findings) == 1                                       # a line that already has a card is not doubled


def test_the_safety_step_failing_to_name_the_ingredient_does_not_keep_a_confirmed_card():
    unnamed = card(checks=[{"type": "ingredient", "status": "needs_input", "attention": True, "detail": {}}])
    kept, settled = settle_ingredient_claims([unnamed], [AD_LINE, AD_LIST], [])
    assert kept == [] and settled and "히알루론산" in settled[0]["reason"]


def test_a_known_ingredient_the_extraction_missed_is_still_checked():
    from capstone_agent.ingredient_presence import names_of
    only_one = card(checks=[{"type": "ingredient", "status": "not_listed", "attention": False, "detail": {"ingredient": "히알루론산"}}])
    assert names_of(only_one) == ["히알루론산", "해양심층수"]
    kept, settled = settle_ingredient_claims([only_one], [AD_LINE, {"id": "t2", "text": "정제수, 소듐하이알루로네이트"}], [])
    assert settled == [] and "해양심층수" in kept[0]["checks"][-1]["summary"]   # one of the two is not in the material


def test_a_named_raw_material_is_looked_up_even_when_the_safety_step_names_nothing():
    from capstone_agent.ingredient_presence import names_of
    assert names_of(card("모란 추출물", checks=[])) == ["모란 추출물"]
    assert names_of(card("셀 소스™ 성분을 함유한 에센스", checks=[])) == ["셀 소스"]
    kept, _ = settle_ingredient_claims([card("셀 소스 성분을 함유한 에센스", checks=[])], [AD_LINE, AD_LIST], [])
    assert kept[0]["checks"][-1]["status"] == "not_in_attachments"          # looked up, and not in the material


def test_copy_that_only_introduces_ingredients_by_name_is_not_a_claim():
    for phrase in ("모란 추출물", "피부 보습 및 보호 성분 판테놀, 알란토인"):
        kept, settled = settle_ingredient_claims([card(phrase, checks=[])], [AD_LINE], [])
        assert kept == [] and settled[0]["source"] == "ingredient_named"


def test_effect_copy_the_first_pass_never_listed_becomes_a_card():
    from capstone_agent.ingredient_presence import rescue_ingredient_claims
    texts = [{"id": "a", "role": "claim", "text": "비타민 콤플렉스로 피부를 보호해주는 항산화 앰플"},
             {"id": "b", "role": "claim", "text": "더 퍼스트 앰플 마스크"},
             {"id": "c", "role": "condition", "text": "10% 할인"},
             {"id": "d", "role": "claim", "text": "필러 침투"}]
    dismissed = [{"id": "x", "textId": "d", "phrase": "필러 침투", "source": "model"}]
    findings, left = rescue_ingredient_claims([], dismissed, texts)
    assert sorted(f["textId"] for f in findings) == ["a", "d"] and left == []


def test_a_brands_own_trademarked_material_is_not_checked_against_the_ingredient_list():
    kept, settled = settle_ingredient_claims([card("셀 소스™ 성분을 함유한 필수 퍼스트 워터 에센스", checks=[])], [AD_LINE, AD_LIST], [])
    assert kept == [] and settled[0]["source"] == "brand_material"           # "퍼스트 워터 에센스" is the product
    kept, _ = settle_ingredient_claims([card(checks=None)], [AD_LINE], [])          # an ordinary name is still looked up
    assert len(kept) == 1
