from types import SimpleNamespace

from capstone_agent.overreach import OverreachReview, review_overreach

LOC = {"tileId": "a:p1:t2"}
TEXTS = [{"id": "t1", "text": "7", "kind": "table", "location": LOC},
         {"id": "t2", "text": "보습 12.3% 개선", "kind": "table", "location": LOC},
         {"id": "t3", "text": "적음 보통 많음", "kind": "copy", "location": LOC}]
CARDS = [{"id": "f1", "phrase": "7", "textId": "t1", "location": LOC, "reason": "수치"},
         {"id": "f2", "phrase": "12.3%", "textId": "t2", "location": LOC, "reason": "수치"}]


class Model:
    client = object()

    def __init__(self, fail=False):
        self.fail, self.seen = fail, []

    async def parse(self, prompt, schema, **kwargs):
        self.seen.append(kwargs["text"])
        if self.fail:
            raise RuntimeError("down")
        return OverreachReview(verdicts=[{"id": "f1", "release": True, "reason": "척도 눈금 번호"},
                                         {"id": "f2", "release": False, "reason": "효능 수치"}])


async def test_released_cards_leave_with_a_reason_and_the_reviewer_sees_the_section():
    model = Model()
    kept, released, warnings = await review_overreach(model, CARDS, TEXTS)
    assert [f["id"] for f in kept] == ["f2"]
    assert released == [{"id": "f1", "phrase": "7", "textId": "t1", "reason": "척도 눈금 번호", "source": "overreach_review"}]
    assert warnings == [] and "적음 보통 많음" in model.seen[0]


async def test_a_failed_review_keeps_every_card():
    kept, released, warnings = await review_overreach(Model(fail=True), CARDS, TEXTS)
    assert kept == CARDS and released == [] and warnings == ["OVERREACH_REVIEW_FAILED"]


async def test_without_a_model_nothing_is_released():
    kept, released, _ = await review_overreach(SimpleNamespace(client=None), CARDS, TEXTS)
    assert kept == CARDS and released == []


async def test_a_section_stays_in_one_batch_and_each_card_sees_the_others():
    other = {"tileId": "a:p1:t9"}
    cards = [{"id": f"x{i}", "phrase": f"문구{i}", "location": other} for i in range(11)] + CARDS
    model = Model()
    await review_overreach(model, cards, TEXTS)
    assert len(model.seen) == 2 and "같은 구간의 다른 카드: 12.3%" in model.seen[1]


class ReleaseAll:
    client = object()

    async def parse(self, prompt, schema, **kwargs):
        ids = [line[1:line.index("]")] for line in kwargs["text"].splitlines() if line.startswith("[")]
        return OverreachReview(verdicts=[{"id": i, "release": True, "reason": "연상한 효능"} for i in ids])


def card(i, phrase, claim_type="exaggeration"):
    return {"id": i, "phrase": phrase, "claimType": claim_type, "textId": "t1", "location": LOC, "reason": "-"}


async def test_cards_that_state_something_to_check_are_never_released_whatever_the_model_answers():
    protected_cards = [card("a", "미백", "whitening"), card("b", "미백 효과"), card("c", "4배 빠른 저분자!"),
                       card("d", "볼륨 개선도 92.7%"), card("e", "[국내최초] 탄력 재생", "absolute"), card("f", "특허 제 0440357 호", "patent"),
                       card("g", "인체적용시험 완료"), card("h", "주름 개선", "wrinkle"), card("i", "단 5회 만으로 리프팅"), card("m", "딱 1분만에 촉촉"),
                       card("j", "자외선 차단"), card("k", "질병 치료", "drug"), card("l", "임상 결과")]
    kept, released, warnings = await review_overreach(ReleaseAll(), protected_cards, TEXTS)
    assert [f["id"] for f in kept] == list("abcdefghimjkl") and released == []
    assert warnings == ["OVERREACH_RELEASE_BLOCKED"]


async def test_cards_without_such_a_claim_are_still_released():
    plain = [card("p", "7", "clinical"), card("q", "추천피부"), card("r", "17겹 수분 케어"), card("s", "촉촉한 사용감"),
             card("t", "제품명 전성분 사용방법")]
    kept, released, warnings = await review_overreach(ReleaseAll(), plain, TEXTS)
    assert kept == [] and [r["id"] for r in released] == list("pqrst") and warnings == []


async def test_protected_and_plain_cards_in_the_same_batch_are_split_by_the_guard():
    cards = [card("a", "미백", "whitening"), card("p", "촉촉한 사용감"), card("c", "보습 12.3% 개선", "clinical")]
    kept, released, warnings = await review_overreach(ReleaseAll(), cards, TEXTS)
    assert [f["id"] for f in kept] == ["a", "c"] and [r["id"] for r in released] == ["p"]
    assert warnings == ["OVERREACH_RELEASE_BLOCKED"]
