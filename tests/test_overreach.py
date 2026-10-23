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
