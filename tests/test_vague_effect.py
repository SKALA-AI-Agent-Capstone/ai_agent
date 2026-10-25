"""Generic care copy that names no specific effect leaves the review; copy that names something checkable stays."""
from capstone_agent.overreach import protected
from capstone_agent.vague_effect import dismiss_vague_effects, effect_claim, vague


def card(phrase, claim_type="exaggeration", i="f"):
    return {"id": i, "phrase": phrase, "claimType": claim_type, "textId": "t"}


def test_care_copy_without_a_named_effect_is_vague():
    for phrase in ["영양이 빈틈없이 흡수되어", "피부 결을 개선시켜", "피부 밸런스를 정돈해주는 영양 수액", "잠든 피부 빛을 깨워",
                   "탄탄하고 건강한 피부로", "주요 활력인자 29종에 도움", "위아래 맞춤형 에센스 설계로 효과를 높이는 시트 마스크",
                   "10~15분 후 마스크를 떼어내고"]:
        assert vague(card(phrase)), phrase


def test_copy_that_names_something_checkable_stays():
    for phrase in ["볼륨 개선도 92.7%", "4배 빠른 저분자!", "볼꺼짐 탄력섬유재생!", "2주간 사용 후 주름 개선", "피부 장벽 강화",
                   "투명한 피부로 가꾸어주는 브라이트닝 세럼",
                   "비타민으로 피부를 보호해주는 항산화 앰플", "[국내최초] 탄력 재생", "특허 받은 성분", "사용 즉시 피부가 20년 젊어집니다"]:
        assert not vague(card(phrase)), phrase
    assert not vague(card("특허청 등록 제 0440357 호", "patent")) and not vague(card("피부 재생", "drug"))
    assert not vague(card("히알루론산 듬뿍", "ingredient"))                   # a named ingredient goes to ingredient_presence
    assert not vague(card("화이트닝 효과", "whitening"))


def test_a_type_stands_only_when_the_words_back_it():
    assert vague(card("외부의 유해 자극으로부터 피부를 보호해주고", "drug"))    # protecting the skin is what cosmetics do
    assert vague(card("피부 결을 개선시켜", "patent"))
    assert not vague(card("4배 빠른 저분자! 필러 침투", "drug"))


def test_expressions_that_only_sound_strong_are_vague():
    assert vague(card("영원히 빛나는 아름다움의 시작", "absolute"))          # a figure of speech, not a guarantee
    assert vague(card("보습 및 탄력을 강화하여 생기를 가꿔주는 고영양 크림"))
    assert not vague(card("업계 최초 탄력 크림", "absolute")) and not vague(card("국내 유일", "absolute"))
    assert not vague(card("평생 보습 지속", "absolute"))                        # a lasting effect is still a claim


def test_a_functional_or_ingredient_type_without_words_to_back_it_is_vague():
    assert vague(card("잠든 피부 빛을 깨워 강렬하게 빛나는 포르테 광채 집중 케어 마스크", "whitening"))
    assert vague(card("주요 활력인자 29종에 도움", "ingredient"))


def test_vague_cards_leave_with_a_reason_and_the_rest_stay_in_order():
    kept, dismissed = dismiss_vague_effects([card("피부 결을 개선시켜", i="a"), card("볼륨 개선도 92.7%", i="b"),
                                             card("잠든 피부 빛을 깨워", i="c")])
    assert [f["id"] for f in kept] == ["b"] and [d["id"] for d in dismissed] == ["a", "c"]
    assert all(d["source"] == "vague_effect" and d["reason"] for d in dismissed)


def test_a_usage_time_is_not_protected_but_a_result_after_a_period_is():
    assert not protected(card("10~15분 후 마스크를 떼어내고"))
    assert not protected(card("2주간 꾸준히 사용하세요"))
    assert protected(card("단 5회 만으로 리프팅"))
    assert protected(card("2주 후 주름 개선"))


def test_cards_rated_low_by_the_first_pass_go_no_further_unless_they_name_something_checkable():
    from capstone_agent.vague_effect import set_aside_low_risk
    cards = [{**card("피부 결을 개선시켜", i="a"), "parseRisk": "low"},
             {**card("볼륨 개선도 92.7%", "clinical", i="b"), "parseRisk": "low"},          # a figure: sent on anyway
             {**card("미백", "whitening", i="c"), "parseRisk": "low"},
             {**card("히알루론산 듬뿍", "ingredient", i="d"), "parseRisk": "low"},
             {**card("잠든 피부를 깨워", i="e"), "parseRisk": "medium"},                    # not low: the later steps decide
             card("촉촉한 사용감", i="f"),
             {**card("모란 추출물 함유", i="g"), "parseRisk": "low"},
             {**card("다마스크 로즈 워터", i="h"), "parseRisk": "low"}]                     # a bare name: set aside                       # an ingredient claim: checked later                                                   # no rating (older result): sent on
    sent, aside = set_aside_low_risk(cards)
    assert [f["id"] for f in sent] == ["b", "c", "d", "e", "f", "g"]
    assert [a["id"] for a in aside] == ["a", "h"] and aside[0]["source"] == "low_at_parse"


def test_the_first_pass_risk_defaults_to_medium_for_older_answers():
    from capstone_shared.review_contracts import TextIssue
    assert TextIssue(phrase="p", reason="r", claimType="exaggeration", rewrites=[]).risk == "medium"


def test_a_skin_concern_is_an_effect_only_with_an_effect_verb():
    assert vague(card("피지 분비가 많은 T존에 맞춘 시트로 청량하면서 끈적임 없이 빠르게 흡수되는"))
    assert vague(card("각질을 부드럽게 정돈하는 효소 성분"))
    assert not vague(card("노폐물 제거, 모공 타이트닝, 과잉 피지 조절"))
    assert not vague(card("실리카 각질 제거 효과, 피지 흡착, 모공 수축"))


def test_a_long_explanation_or_seller_notice_is_not_a_claim_unless_it_states_an_effect():
    kurly = ("컬리는 화장품, 샴푸, 치약, 세제 등 인체에 닿는 리빙 제품의 전성분을 공개하고 성분의 위험 정도를 EWG 기준에 따라 "
             "1~10 숫자로 안내하고 있습니다. EWG가 세계적으로 널리 쓰이는 척도인 것은 맞지만 안전성을 증명하는 절대적인 기준은 "
             "아니에요. 이에 컬리는 식품의약품안전처, 에코서트, 미국식품의약국, 미국화장품원료검토위원회 등 신뢰 있는 인증기관의 "
             "의견을 참고하여 유해성 여부를 꼼꼼하게 따져 보고 있습니다.")
    kept, dismissed = dismiss_vague_effects([card(kurly, "ingredient", i="k")])
    assert kept == [] and dismissed[0]["source"] == "notice_text"
    long_claim = card("임상 시험 결과 " + "피부 탄력이 좋아지고 " * 12 + "주름이 30% 개선되었습니다", "clinical", i="l")
    assert dismiss_vague_effects([long_claim])[0] == [long_claim]


def test_an_effect_claim_the_judge_dropped_comes_back_as_a_card():
    from capstone_agent.ingredient_presence import rescue_ingredient_claims
    texts = [{"id": "t1", "text": "피부 속에 노폐물과 오래된 각질을 매끄럽게 제거하고 진정시켜", "location": {"fileId": "a"}},
             {"id": "t2", "text": "촉촉한 피부를 유지하는데 도움을 줍니다", "location": {"fileId": "a"}}]
    dismissed = [{"id": "d1", "phrase": texts[0]["text"], "textId": "t1", "source": "model"},
                 {"id": "d2", "phrase": texts[1]["text"], "textId": "t2", "source": "model"},
                 {"id": "d3", "phrase": texts[0]["text"], "textId": "t1", "source": "rule_sweep"}]
    findings, left = rescue_ingredient_claims([], dismissed, texts)
    assert [f["textId"] for f in findings] == ["t1"] and findings[0]["source"] == "effect_claim"
    assert [d["id"] for d in left] == ["d2", "d3"]


def test_the_overreach_review_cannot_release_an_effect_on_a_skin_concern():
    assert protected(card("노폐물 제거, 모공 타이트닝, 각질 케어 조절"))
    assert not protected(card("피지 분비가 많은 T존에 맞춘 시트"))


def test_a_named_effect_such_as_antioxidation_is_an_effect_claim():
    assert effect_claim("비타민 콤플렉스로 피부를 보호해주는 항산화 앰플")
    assert not effect_claim("피부에 영양을 채워 균형 있게 가꿔주는 에센셜 에멀전")


def test_immediacy_alone_is_pace_but_an_immediate_change_is_specific():
    assert vague({"phrase": "즉각적으로 피부에 풍부한 수분과 영양 성분을 공급해주어 촉촉", "claimType": "exaggeration"})
    assert not vague({"phrase": "사용 즉시 피부가 젊어집니다", "claimType": "exaggeration"})


def test_lifting_as_mood_is_vague_but_a_lifting_effect_is_not():
    assert vague({"phrase": "탄력 넘치는 리프팅 시너지 조합!", "claimType": "exaggeration"})
    assert not vague({"phrase": "사용 후 즉각적인 리프팅 효과", "claimType": "exaggeration"})
