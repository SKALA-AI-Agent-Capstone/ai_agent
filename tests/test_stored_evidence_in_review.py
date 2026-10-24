"""A product's evidence is registered once and read once; a review must treat it as submitted evidence. Counting only the
files sent with the review made the model skip the tool and marked every figure claim 'no evidence submitted'."""
from types import SimpleNamespace

import capstone_agent.review_engine as engine
from capstone_agent.checks import cross_checks
from capstone_agent.review_engine import evidence_total, select_business_tools


def request(sent=0, stored=0):
    assets = [SimpleNamespace(role="evidence", model_dump=lambda: {"id": "e", "role": "evidence"}) for _ in range(sent)]
    assets.append(SimpleNamespace(role="advertisement", model_dump=lambda: {"id": "a", "role": "advertisement"}))
    return SimpleNamespace(assets=assets, productEvidenceCount=stored)


def test_evidence_counts_what_was_sent_and_what_the_product_already_holds():
    assert evidence_total(request()) == 0
    assert evidence_total(request(sent=2)) == 2
    assert evidence_total(request(stored=1)) == 1
    assert evidence_total(request(sent=1, stored=2)) == 3
    assert evidence_total(None) == 0
    assert evidence_total(SimpleNamespace(assets=[])) == 0          # an older request without the field


async def test_the_model_is_told_about_the_products_evidence_when_nothing_was_sent(monkeypatch):
    seen = []

    class Provider:
        client = True

        async def call_with_tools(self, messages, tools):
            seen.append(messages[-1]["content"])
            return SimpleNamespace(content="", tool_calls=[])

    state = {"tenantId": "t", "reviewId": "r", "product": {"name": "수분크림"}, "texts": [{"text": "보습 20% 개선"}],
             "findings": [{"phrase": "보습 20% 개선", "reason": "수치", "claimType": "moisture"}]}
    await select_business_tools(Provider(), state, request(stored=2))
    assert "제출된 증빙서류: 2건" in seen[0] and "제품에 등록된 증빙 2건" in seen[0]
    seen.clear()
    await select_business_tools(Provider(), state, request())
    assert "제출된 증빙서류: 0건" in seen[0]


async def test_a_figure_claim_is_not_reported_as_missing_evidence_when_the_product_has_evidence():
    finding = {"id": "f1", "phrase": "보습 20% 개선", "claimType": "moisture", "requiredChecks": ["substantiation"],
               "evidenceRequirements": [{"expression": "수치", "requirement": "시험 자료", "source": "화장품법 제14조"}]}

    async def call(server, tool, arguments):
        return {"status": "not_found", "items": []}

    async def checks(count):
        out = await cross_checks([dict(finding)], product={"name": "수분크림"}, evidence={"status": "extracted"},
                                 evidence_count=count, ad_text="보습 20% 개선", provider=object(),
                                 review_date="2026-10-23", call=call)
        return {c["type"]: c["status"] for c in out["findings"][0]["checks"]}

    assert (await checks(evidence_total(request())))["substantiation"] == "evidence_missing"
    assert (await checks(evidence_total(request(stored=1))))["substantiation"] == "evidence_submitted"


class Message:
    def __init__(self, content="", names=()):
        self.content = content
        self.tool_calls = [SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=n),
                                           model_dump=lambda n=n, i=i: {"id": f"c{i}", "type": "function",
                                                                        "function": {"name": n, "arguments": "{}"}})
                           for i, n in enumerate(names)] or None


def provider(*turns):
    replies = iter(turns)

    class Provider:
        client = True

        async def call_with_tools(self, messages, tools):
            return next(replies, Message())

    return Provider()


STATE = {"tenantId": "t", "reviewId": "r", "product": {"name": "수분크림"}, "texts": [{"text": "촉촉한 사용감"}],
         "findings": [{"phrase": "촉촉한 사용감", "reason": "감성 표현", "claimType": "exaggeration", "requiredChecks": []}]}


async def test_a_skipped_evidence_tool_says_the_model_found_it_unnecessary_and_why(monkeypatch):
    async def mcp(name, arguments, request=None):
        return {"status": "found"}
    monkeypatch.setattr(engine, "mcp_call", mcp)
    reason = "수치나 시험 결과를 말하는 문구가 없어 증빙 대조가 필요 없습니다."
    *_, trace = await select_business_tools(provider(Message(names=["verify_functional_approval"]), Message(reason)),
                                            STATE, request(stored=1))
    assert {"tool": "structure_evidence", "called": False, "reason": "MODEL_NOT_NEEDED", "note": reason} in trace
    assert {"tool": "verify_functional_approval", "called": True} in trace


async def test_without_registered_evidence_the_tool_is_marked_as_having_nothing_to_read(monkeypatch):
    async def mcp(name, arguments, request=None):
        return {"status": "found"}
    monkeypatch.setattr(engine, "mcp_call", mcp)
    *_, trace = await select_business_tools(provider(Message("필요한 도구 없음")), STATE, request())
    assert {"tool": "structure_evidence", "called": False, "reason": "NO_EVIDENCE"} in trace
    nothing = {**STATE, "product": {}}
    *_, trace = await select_business_tools(provider(), nothing, request())
    assert {t["tool"]: t["reason"] for t in trace} == {"structure_evidence": "NO_EVIDENCE",
                                                         "verify_functional_approval": "NO_PRODUCT_NAME",
                                                         "verify_patent": "NO_PATENT_CLAIM"}


async def test_without_a_model_every_available_tool_runs_and_the_rest_say_why_not(monkeypatch):
    async def mcp(name, arguments, request=None):
        return {"status": "found"}
    monkeypatch.setattr(engine, "mcp_call", mcp)
    *_, trace = await select_business_tools(SimpleNamespace(client=None), STATE, request(stored=1))
    assert {"tool": "structure_evidence", "called": True} in trace
    assert {"tool": "verify_patent", "called": False, "reason": "NO_PATENT_CLAIM"} in trace
