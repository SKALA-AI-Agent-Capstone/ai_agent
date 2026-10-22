import json
from types import SimpleNamespace

from capstone_agent import judged_basis as jb
from capstone_agent.judged_basis import BasisCheck, ClaimJudgments, judged_basis

DAY = "2026-10-22"


def hit(section, excerpt, key="legalize-kr:화장품법/법률", page=None):
    return {"source": "화장품법", "version": "v1", "scope": "화장품법", "excerpt": excerpt, "similarity": 0.6,
            "source_type": "law", "document_key": key, "section": section, "page": page, "source_url": "u",
            "effective_from": "2026-01-01", "retrieval": "search", "item": None, "related": []}


SUBSTANTIATION = hit("제14조 (표시ㆍ광고 내용의 실증 등)", "① 영업자는 자기가 행한 표시ㆍ광고 중 사실과 관련한 사항에 대하여는 실증할 수 있어야 한다.")
GENERAL = hit("제1조 (목적)", "이 법은 화장품의 제조ㆍ수입ㆍ판매 및 수출 등에 관한 사항을 규정한다.")


def text(tid, value):
    return {"id": tid, "text": value, "location": {"kind": "image", "fileId": "a", "page": 1, "bbox": None}}


def finding(fid, phrase, tid, claim="clinical"):
    return {"id": fid, "phrase": phrase, "start": 0, "end": len(phrase), "textId": tid, "claimType": claim,
            "reason": "1차", "basis": [], "baseBasis": [], "references": [], "location": {}}


def tool_call(name, args, n):
    return SimpleNamespace(id=f"call{n}", function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)),
                           model_dump=lambda: {"id": f"call{n}", "type": "function", "function": {"name": name, "arguments": "{}"}})


class Rag:
    def __init__(self, searches):
        self.searches, self.calls = list(searches), []

    async def __call__(self, server, tool, arguments):
        self.calls.append((tool, arguments))
        if tool == "scan_claim_lines":
            return {"matches": [{"index": 1, "line": "오랫동안 촉촉한 윤기 3중 식물성 오일", "matched": ["3중"],
                                 "rules": ["substantiation"], "labels": ["수치 표현"]}]}
        if tool == "lookup_claim_rules":
            return {"status": "NO_RELEVANT_EVIDENCE", "matches": []}
        if tool == "search_regulations":
            return {"status": "FOUND", "hits": self.searches.pop(0) if self.searches else []}
        return {"status": "FOUND", "hits": []}


class Model:
    """Judges, runs one search per loop, and verifies with whatever quote the test sets."""
    client = True

    def __init__(self, verdicts):
        self.verdicts, self.loops, self.seen = list(verdicts), 0, []

    async def parse(self, prompt, schema, **kwargs):
        self.seen.append((schema.__name__, kwargs["text"]))
        if schema is ClaimJudgments:
            return ClaimJudgments(judgments=[
                {"id": "f1", "problem": True, "claimType": "clinical", "span": "임상으로 입증", "purpose": "시험 결과 표시의 실증 요건",
                 "reason": "임상 결과를 효능 근거로 내세움"},
                {"id": "rule-1", "problem": False, "claimType": "exaggeration", "span": "", "purpose": "",
                 "reason": "3중은 오일 구성 설명이고 효능 주장이 아님"}])
        assert schema is BasisCheck
        return BasisCheck(verdicts=self.verdicts.pop(0))

    async def call_with_tools(self, messages, tools):
        self.loops += 1
        if messages[-1]["role"] == "tool":
            return SimpleNamespace(tool_calls=[], content="충분함")
        return SimpleNamespace(tool_calls=[tool_call("search_regulations", {"query": "시험 결과 표시 실증"}, self.loops)],
                               content="")


async def test_judges_first_then_keeps_only_basis_the_verifier_quoted():
    texts = [text("t1", "임상으로 입증된 주름 개선"), text("t2", "오랫동안 촉촉한 윤기 3중 식물성 오일")]
    rag = Rag([[SUBSTANTIATION, GENERAL]])
    model = Model([[{"id": "c1", "applies": True, "quote": "실증할 수 있어야 한다", "why": "시험 결과는 실증 자료가 있어야 함"},
                    {"id": "c2", "applies": True, "quote": "지어낸 문장", "why": "x"}]])
    out = await judged_basis([finding("f1", "임상으로 입증된 주름 개선", "t1")], model, texts=texts, call=rag)

    # The rule sweep's '3중' line is judged not a concern and leaves the findings.
    assert [d["phrase"] for d in out["dismissed"]] == ["오랫동안 촉촉한 윤기 3중 식물성 오일"]
    [card] = out["findings"]
    assert card["problemSpan"] == {"text": "임상으로 입증", "start": 0, "end": 7, "bbox": None}
    # The model wrote the query; the code filled in the date and the size.
    search = next(a for t, a in rag.calls if t == "search_regulations")
    assert search["query_text"] == "시험 결과 표시 실증" and search["review_date"]
    # A quote that is not in the candidate text is rejected, so only the substantiation article remains.
    assert [b["article"] for b in card["basis"]] == ["제14조 (표시ㆍ광고 내용의 실증 등)"]
    assert card["basis"][0]["verification"]["quote"] == "실증할 수 있어야 한다"
    assert card["legalSearch"] == "verified" and out["status"] == "connected"
    # The verifier sees the phrase and the candidates, never the judgment or its purpose.
    verify_text = next(t for name, t in model.seen if name == "BasisCheck")
    assert "시험 결과 표시의 실증 요건" not in verify_text and "임상 결과를 효능 근거로" not in verify_text


async def test_retries_once_with_the_rejected_candidates_then_reports_no_verified_basis():
    texts = [text("t1", "임상으로 입증된 주름 개선")]
    rag = Rag([[GENERAL], [hit("제2조 (정의)", "화장품이란 인체를 청결ㆍ미화하여")]])
    model = Model([[{"id": "c1", "applies": False, "quote": "", "why": "목적 조항"}],
                   [{"id": "c2", "applies": False, "quote": "", "why": "정의 조항"}]])
    out = await judged_basis([finding("f1", "임상으로 입증된 주름 개선", "t1")], model, texts=texts, call=rag)
    [card] = out["findings"]
    assert card["basis"] == [] and card["legalSearch"] == "not_verified"
    assert card["basisTrace"]["retried"] is True and card["basisTrace"]["candidates"] == 2
    second_loop = [t for name, t in model.seen if name == "BasisCheck"]
    assert len(second_loop) == 2 and "제1조" not in second_loop[1]  # only new candidates are verified again


async def test_tool_loop_stops_at_the_guardrail():
    class Greedy(Model):
        async def call_with_tools(self, messages, tools):
            self.loops += 1
            return SimpleNamespace(tool_calls=[tool_call("search_regulations", {"query": f"q{self.loops}"}, self.loops)], content="")

    rag = Rag([[GENERAL]] * 10)
    model = Greedy([[{"id": "c1", "applies": False, "quote": "", "why": "-"}], []])
    found = jb.Candidates()
    trace = await jb.gather(model, rag, finding("f1", "a", "t1"), "a", DAY, found)
    searches = [t for t in trace if t.get("tool") == "search_regulations"]
    assert len(searches) == jb.MAX_SAME_BASIS_TOOL


def test_problem_span_box_is_the_union_of_the_matching_ocr_words():
    block = {"words": [{"text": "임상으로", "bbox": [0.1, 0.1, 0.2, 0.12]}, {"text": "입증된", "bbox": [0.21, 0.1, 0.3, 0.12]},
                       {"text": "주름", "bbox": [0.31, 0.1, 0.36, 0.12]}]}
    assert jb.span_box("임상으로 입증", block) == [0.1, 0.1, 0.3, 0.12]
    assert jb.span_box("미백", block) is None
