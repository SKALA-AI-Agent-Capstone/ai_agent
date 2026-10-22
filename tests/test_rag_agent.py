from langgraph.checkpoint.memory import InMemorySaver

from capstone_agent.checks import claimed_functions, cross_checks, functional_check
from capstone_agent.regulations import build_queries, legal_basis

DAY = "2026-10-21"


def hit(section, similarity=1.0, retrieval="rule", item=None, **extra):
    return {"source": "화장품법", "version": "2026-10-20 제22113호", "scope": "화장품법 > 제3장", "excerpt": f"{section} 원문",
            "similarity": similarity, "trust_level": "high", "source_type": "law", "document_key": "legalize-kr:화장품법/법률",
            "section": section, "page": None, "source_url": "https://github.com/legalize-kr/legalize-kr", "acquisition": "legalize-kr",
            "effective_from": "2026-10-20", "effective_until": None, "retrieval": retrieval, "item": item, "related": [], **extra}


def finding(phrase, claim_type="drug", fid=None):
    return {"id": fid or phrase, "phrase": phrase, "claimType": claim_type, "reason": "합성", "basis": [], "baseBasis": [],
            "title": "근거 대조 전 확인할 문구", "rewrites": [], "references": []}


def fake_rag(rules=None, search=None, cases=None, fail=False, empty=False, scan=None):
    async def call(server, tool, arguments):
        if fail:
            raise ConnectionError("rag-mcp down")
        if tool == "scan_claim_lines":
            return {"matches": scan or []}
        if empty:
            return {"status": "CORPUS_NOT_INDEXED", "matches": [], "hits": []}
        if tool == "lookup_claim_rules":
            return {"status": "FOUND" if rules else "NO_RELEVANT_EVIDENCE", "matches": rules or []}
        if tool == "search_regulations":
            return {"status": "FOUND" if search else "NO_RELEVANT_EVIDENCE", "hits": search or []}
        return {"status": "FOUND" if cases else "CORPUS_NOT_INDEXED", "hits": cases or []}
    return call


DRUG_RULE = {"rule_id": "drug", "label": "의약품 오인 표현", "matched_by": "claim_type", "matched_text": "drug", "checks": [],
             "missing": [], "evidence_requirements": [],
             "provisions": [hit("제13조 (부당한 표시ㆍ광고 행위 등의 금지)", item="제1항제1호",
                                related=[{"source": "화장품법 시행규칙", "section": "제22조 (표시ㆍ광고의 범위 등)", "excerpt": "별표 5와 같다",
                                          "version": "v", "relation": "implemented_by", "document_key": "k", "source_url": "u"}]),
                            hit("별표 5 제2호 가목", document_key="lawapi:S06")]}


def test_supplementary_query_keeps_statute_wording():
    [query] = build_queries(finding("아토피 치료 크림", "drug"))
    assert query.startswith("아토피 치료 크림") and "의약품으로 잘못 인식할 우려" in query
    assert len(build_queries(finding("피부 주름 90% 감소", "wrinkle"))) == 2


async def test_rule_provisions_come_first_with_item_and_related_chain():
    lookup = await legal_basis([finding("아토피 치료 크림")], call=fake_rag(rules=[DRUG_RULE]))
    assert lookup["status"] == "connected"
    basis = lookup["findings"][0]["basis"]
    assert basis[0]["article"] == "제13조 (부당한 표시ㆍ광고 행위 등의 금지) 제1항제1호"
    assert basis[0]["retrieval"] == "rule" and basis[0]["ruleId"] == "drug"
    assert basis[0]["related"][0]["relation"] == "implemented_by"
    assert basis[1]["sourceId"] == "lawapi:S06"
    assert lookup["findings"][0]["rules"][0]["id"] == "drug"
    assert lookup["findings"][0]["legalSearch"] == "basis_found"


async def test_search_does_not_repeat_an_article_the_rule_already_cites():
    same = hit("제13조 (부당한 표시ㆍ광고 행위 등의 금지)", similarity=0.7, retrieval="search")
    lookup = await legal_basis([finding("아토피 치료 크림")], call=fake_rag(rules=[DRUG_RULE], search=[same]))
    articles = [b["article"] for b in lookup["findings"][0]["basis"]]
    assert articles.count("제13조 (부당한 표시ㆍ광고 행위 등의 금지)") == 0 and len(articles) == 2


async def test_weak_search_hits_do_not_become_basis():
    lookup = await legal_basis([finding("촉촉한 사용감", "exaggeration")],
                               call=fake_rag(search=[hit("제2조 (정의)", similarity=0.16, retrieval="search")]))
    assert lookup["findings"][0]["basis"] == [] and lookup["findings"][0]["legalSearch"] == "no_match"


async def test_every_finding_is_searched_without_a_count_cap():
    findings = [finding(f"문구 {i}", fid=str(i)) for i in range(60)]
    lookup = await legal_basis(findings, call=fake_rag(rules=[DRUG_RULE]))
    assert all(f["legalSearch"] == "basis_found" for f in lookup["findings"])
    assert not any("LIMITED" in w for w in lookup["warnings"])


async def test_empty_corpus_and_outage_are_distinct():
    assert (await legal_basis([finding("a")], call=fake_rag(empty=True)))["status"] == "not_indexed"
    failed = await legal_basis([finding("a")], call=fake_rag(fail=True))
    assert failed["status"] == "error" and failed["findings"][0]["basis"] == []
    assert (await legal_basis([], call=fake_rag()))["status"] == "skipped"


def test_claimed_functions_are_read_from_the_phrase():
    assert claimed_functions("눈가 주름 12.08% 개선") == ["주름개선"]
    assert claimed_functions("기미 잡티 브라이트닝") == ["미백"]
    assert claimed_functions("촉촉한 보습") == []


def test_functional_check_compares_claim_with_reported_functions():
    reports = {"status": "found", "exact": [{"productName": "A크림", "functions": ["미백"], "cancelled": False}], "candidates": []}
    assert functional_check(finding("주름 개선", "wrinkle"), reports, "A크림")["status"] == "mismatch"
    assert functional_check(finding("기미 개선", "whitening"), reports, "A크림")["status"] == "consistent"
    assert functional_check(finding("주름 개선", "wrinkle"), {"status": "not_found", "exact": [], "candidates": []}, "B")["status"] == "not_found"
    error = functional_check(finding("주름 개선", "wrinkle"), {"status": "error", "error": "API_NOT_CONFIGURED"}, "B")
    assert error["status"] == "unavailable" and error["attention"] is True


async def test_cross_checks_run_only_what_the_rules_ask_for():
    calls = []

    async def call(server, tool, arguments):
        calls.append(tool)
        if tool == "find_functional_reports":
            return {"status": "found", "exact": [{"productName": "A크림", "functions": ["미백"], "cancelled": False,
                                                  "manufacturer": "가나화장품"}], "candidates": []}
        if tool == "verify_gmp_company":
            return {"status": "not_found", "items": []}
        if tool == "check_ingredient_standard":
            return {"status": "prohibited" if arguments["ingredient"] == "히드로퀴논" else "not_listed", "listings": []}
        return {"status": "not_found", "items": []}

    findings = [{**finding("주름 90% 개선", "wrinkle", "f1"), "requiredChecks": ["functional_report", "substantiation"],
                 "evidenceRequirements": [{"expression": "피부노화 완화", "requirement": "인체 적용시험 자료 제출", "source": "실증 규정"}]},
                {**finding("히드로퀴논 함유", "ingredient", "f2"), "requiredChecks": ["ingredient"]},
                {**finding("CGMP 시설 생산", "exaggeration", "f3"), "requiredChecks": ["gmp"]},
                {**finding("촉촉한 보습", "exaggeration", "f4"), "requiredChecks": []}]
    out = await cross_checks(findings, product={"name": "A크림"}, evidence={"status": "not_submitted"}, evidence_count=0,
                             ad_text="", provider=object(), review_date=DAY, call=call)
    by_id = {f["id"]: {c["type"]: c["status"] for c in f["checks"]} for f in out["findings"]}
    assert by_id["f1"] == {"functional_report": "mismatch", "substantiation": "evidence_missing"}
    assert by_id["f2"] == {"ingredient": "prohibited"}
    assert by_id["f3"] == {"gmp": "gmp_not_listed"}
    assert by_id["f4"] == {}
    assert calls.count("find_functional_reports") == 1


async def test_graph_runs_legal_basis_then_cross_checks_and_rewrites_with_the_basis(monkeypatch):
    import capstone_agent.review_engine as engine
    from capstone_agent.review_models import TextAssessment
    from capstone_agent.review_engine import RewriteDraft, RewriteVerification, SeverityAssessment

    async def ocr(native, tiles, provider):
        return [{"id": "t1", "text": "아토피 치료 크림", "location": {"kind": "image", "fileId": "a", "page": 1, "bbox": None}}], []

    async def fake_basis(findings, **kwargs):
        return await legal_basis(findings, call=fake_rag(rules=[DRUG_RULE]))

    async def fake_checks(findings, **kwargs):
        return {"findings": [{**f, "checks": [{"type": "substantiation", "status": "evidence_missing"}]} for f in findings], "summary": {}, "warnings": []}

    monkeypatch.setattr(engine, "extract_text", ocr)
    async def fake_judged(findings, provider, **kwargs):
        return await fake_basis(findings)

    monkeypatch.setattr(engine, "judged_basis", fake_judged)
    monkeypatch.setattr(engine, "cross_checks", fake_checks)
    prompts = []

    class FakeProvider:
        client = True

        async def parse(self, prompt, schema, **kwargs):
            prompts.append(kwargs.get("text", ""))
            if schema is TextAssessment:
                return schema(findings=[{"phrase": "아토피 치료", "reason": "의약품 오인", "claimType": "drug", "rewrites": []}])
            if schema is SeverityAssessment:
                return SeverityAssessment(severity="high", reason="합성 근거와 증빙 검토")
            if schema is RewriteDraft:
                return RewriteDraft(text="민감한 피부를 위한 크림")
            if schema is RewriteVerification:
                return RewriteVerification(passed=True, reasoning="합성")
            raise AssertionError(f"unexpected schema {schema}")

    graph = engine.build_graph([], [], FakeProvider(), InMemorySaver())
    output = await graph.ainvoke({"reviewId": "r", "tenantId": "t", "versionNo": 1, "product": {"name": "A크림"}},
                                 {"configurable": {"thread_id": "t:r:1"}})
    result = output["result"]
    assert result["retrievalStatus"] == "connected"
    assert result["findings"][0]["basis"][0]["article"].endswith("제1항제1호")
    assert result["findings"][0]["checks"] == [{"type": "substantiation", "status": "evidence_missing"}]
    assert result["findings"][0]["severity"] == "high"
    assert result["violations"][0]["severity"] == "high"
    assert any("제13조" in text for text in prompts), "rewrite prompt should carry the retrieved article"
    assert result["findings"][0]["rewrites"] == ["민감한 피부를 위한 크림"]


async def test_rule_sweep_adds_high_risk_lines_the_model_missed():
    texts = [{"id": "t1", "text": "아토피 치료 크림", "location": {"kind": "image"}},
             {"id": "t2", "text": "CGMP 인증 시설에서 생산", "location": {"kind": "image"}}]
    scan = [{"index": 0, "line": "아토피 치료 크림", "rules": ["drug"], "matched": ["치료"]},
            {"index": 1, "line": "CGMP 인증 시설에서 생산", "rules": ["official_endorsement", "gmp"], "matched": ["인증", "GMP"]}]
    lookup = await legal_basis([finding("아토피 치료")], texts=texts, call=fake_rag(rules=[DRUG_RULE], scan=scan))
    phrases = [f["phrase"] for f in lookup["findings"]]
    assert phrases == ["아토피 치료", "CGMP 인증 시설에서 생산"]
    assert lookup["findings"][1]["source"] == "rule_sweep" and lookup["findings"][1]["textId"] == "t2"


async def test_rule_sweep_skips_noisy_duplicates_of_an_existing_claim():
    texts = [{"id": "t1", "text": "4 HS 8.51% 개선", "location": {}}, {"id": "t2", "text": "식약처 심사를 완료하였습니다", "location": {}}]
    scan = [{"index": 0, "line": "4 HS 8.51% 개선", "rules": ["substantiation"], "matched": ["8.51%"]},
            {"index": 1, "line": "식약처 심사를 완료하였습니다", "rules": ["official_endorsement"], "matched": ["식약처"]}]
    lookup = await legal_basis([finding("속 보습 8.51% 개선", "clinical")], texts=texts, call=fake_rag(rules=[DRUG_RULE], scan=scan))
    assert [f["phrase"] for f in lookup["findings"]] == ["속 보습 8.51% 개선", "식약처 심사를 완료하였습니다"]
