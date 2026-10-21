from io import BytesIO
import json
from pathlib import Path
from PIL import Image
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from capstone_shared.documents import Tile, tile_image, prepare_document
from capstone_agent.review_engine import (build_graph, Provider, deduplicate, correct_ocr_text, OcrCorrection,
    tile_manifest, assign_tile)


async def test_long_text_eval_cases_keep_claims_in_every_position(monkeypatch):
    import capstone_agent.review_engine as engine
    from capstone_agent.review_models import TextAssessment, VisualAssessment

    monkeypatch.setenv("MODEL_INPUT_TOKEN_BUDGET", "3000")
    cases = json.loads((Path(__file__).parent / "data/text_review_eval.json").read_text(encoding="utf-8"))
    for case in cases:
        ad = (case.get("prefix", "") + case.get("text", "") * case.get("repeat_before", 0)
              + case.get("insert", "") + case.get("text", "") * case.get("repeat_after", 0)
              + case.get("suffix", ""))
        async def extract(native, tiles, provider):
            return [{"id":"ad", "text":ad, "location":{"fileId":"synthetic", "page":1}}], []
        monkeypatch.setattr(engine, "extract_text", extract)

        class FakeProvider:
            client = True
            seen = []

            async def parse(self, prompt, schema, **kwargs):
                if schema is TextAssessment:
                    chunk = kwargs["text"]
                    self.seen.append(chunk)
                    return TextAssessment(findings=[{"phrase":phrase,"reason":"합성 평가", "claimType":"exaggeration","rewrites":[]}
                        for phrase in set(case["expected"]) for _ in range(chunk.count(phrase))])
                return VisualAssessment(sectionTitle="",findings=[],unreadable=False)

        provider = FakeProvider()
        graph = build_graph([], [], provider, InMemorySaver())
        output = await graph.ainvoke({"reviewId":case["id"],"tenantId":"synthetic","versionNo":1,"product":{}},
                                    {"configurable":{"thread_id":f"synthetic:{case['id']}:1"}})
        assert len(provider.seen) > 1
        assert len(output["result"]["findings"]) == len(case["expected"])
        assert [finding["phrase"] for finding in output["result"]["findings"]] == case["expected"]
        assert all(ad[finding["start"]:finding["end"]] == finding["phrase"] for finding in output["result"]["findings"])
        assert "TEXT_REVIEW_INPUT_TRUNCATED" not in output["result"]["warnings"]


async def test_failed_text_chunk_is_reported_while_later_chunks_continue(monkeypatch):
    import capstone_agent.review_engine as engine
    from capstone_agent.review_models import TextAssessment, VisualAssessment

    monkeypatch.setenv("MODEL_INPUT_TOKEN_BUDGET", "3000")
    ad = "일반적인 사용감.\n" * 500 + "마지막 문구는 모든 피부병을 치료합니다."
    async def extract(native, tiles, provider):
        return [{"id":"ad", "text":ad, "location":{"fileId":"synthetic", "page":1}}], []
    monkeypatch.setattr(engine, "extract_text", extract)

    class FakeProvider:
        client = True
        calls = 0

        async def parse(self, prompt, schema, **kwargs):
            if schema is TextAssessment:
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("synthetic model failure")
                phrase = "모든 피부병을 치료합니다"
                return TextAssessment(findings=[{"phrase":phrase,"reason":"합성 평가","claimType":"drug","rewrites":[]}]
                                      if phrase in kwargs["text"] else [])
            return VisualAssessment(sectionTitle="",findings=[],unreadable=False)

    provider = FakeProvider()
    output = await build_graph([], [], provider, InMemorySaver()).ainvoke(
        {"reviewId":"failure-case","tenantId":"synthetic","versionNo":1,"product":{}},
        {"configurable":{"thread_id":"synthetic:failure-case:1"}})
    result = output["result"]
    assert provider.calls > 1
    assert result["pipelineStatus"]["text"] == "partial"
    assert "TEXT_REVIEW_CHUNK_FAILED:1" in result["warnings"]
    assert [finding["phrase"] for finding in result["findings"]] == ["모든 피부병을 치료합니다"]


async def test_ocr_correction_fixes_broken_spacing_without_changing_line_count():
    class FakeProvider:
        async def parse(self,prompt,schema,**kwargs):
            return OcrCorrection(lines=["7일 주름 개선","SPF50 PA++++"])
    blocks=[{"id":"a","text":"7 일 주 름 개 선","location":{}},{"id":"b","text":"SPF50 PA++++","location":{}}]
    fixed=await correct_ocr_text(FakeProvider(),blocks)
    assert [b["text"] for b in fixed]==["7일 주름 개선","SPF50 PA++++"]
    assert fixed[0]["id"]=="a" and fixed[1]["id"]=="b"  # location/id survive, only text replaced


async def test_ocr_correction_falls_back_to_raw_text_on_line_count_mismatch():
    class BadProvider:
        async def parse(self,prompt,schema,**kwargs):
            return OcrCorrection(lines=["합쳐버린 한 줄"])
    blocks=[{"id":"a","text":"원문 1","location":{}},{"id":"b","text":"원문 2","location":{}}]
    result=await correct_ocr_text(BadProvider(),blocks)
    assert result is blocks


def fixed_tiles(height=3000, size=1600, overlap=240):
    """Overlapping tiles like the former fixed tiler, to test ownership in shared rows."""
    image=Image.new("RGB",(64,height))
    tops=list(range(0,height,size-overlap))
    tops=tops[:next(i for i,t in enumerate(tops) if t+size>=height)+1]
    return [Tile(f"a:p1:t{i}","a","long.png",1,(0,t,64,min(t+size,height)),image.size,None) for i,t in enumerate(tops)]


def striped_page(blocks):
    """A long page of solid colored blocks separated by white gaps, like a product detail page."""
    from PIL import ImageDraw
    height=sum(h for h,_ in blocks)+40*(len(blocks)-1)
    image=Image.new("RGB",(848,height),"white"); draw=ImageDraw.Draw(image); y=0; edges=[]
    for h,color in blocks:
        draw.rectangle((0,y,847,y+h-1),fill=color)
        for line in range(y+60,y+h-60,90):  # text-like stripes inside each block
            draw.rectangle((120,line,700,line+18),fill="black")
        y+=h+40; edges.append(y-20)
    return image, edges[:-1]


def test_long_page_is_cut_between_content_blocks():
    image,gaps=striped_page([(900,"#d9c2a0"),(1300,"#f3e1e1"),(700,"#c0392b"),(1200,"#eeeeee"),(1000,"#2c3e50")])
    tiles=tile_image(image,"synthetic","synthetic.png")
    cuts=[t.box[1] for t in tiles[1:]]
    assert tiles[0].box[1]==0 and tiles[-1].box[3]==image.height
    assert all(a.box[3]==b.box[1] for a,b in zip(tiles,tiles[1:]))  # sections touch, never overlap
    # One cut per gap, each in the blank margin around the gap and never inside a block's text.
    assert len(cuts)==len(gaps) and all(abs(c-g)<=60 for c,g in zip(cuts,gaps))
    assert all(t.location()["bbox"] is None for t in tiles)


def test_short_image_stays_one_section_and_blank_long_image_is_bounded():
    assert len(tile_image(Image.new("RGB",(1190,1684),"white"),"a","page.png"))==1
    tiles=tile_image(Image.new("RGB",(848,19799),"white"),"a","blank.png")
    assert all(320<=t.box[3]-t.box[1]<=1600 for t in tiles) and tiles[-1].box[3]==19799


def test_input_is_decoded_from_bytes_not_filename():
    output=BytesIO();Image.new("RGB",(64,64),"white").save(output,format="PNG")
    blocks,tiles=prepare_document(output.getvalue(),{"id":"a","name":"misleading.pdf"})
    assert blocks==[] and len(tiles)==1


def test_overlap_dedup_does_not_remove_separate_repeated_ad_claims():
    def block(y):
        return {"text":"같은 문구","location":{"fileId":"a","page":1,"bbox":[0,y,1,y+0.1]}}
    assert len(deduplicate([block(0),block(0.05),block(0.8)]))==2


def test_tile_manifest_keeps_coordinates_only():
    tiles=fixed_tiles()
    manifest=tile_manifest(tiles)
    assert [t["id"] for t in manifest]==[t.id for t in tiles]
    assert [t["index"] for t in manifest]==[0,1,2]
    assert manifest[1]=={"id":"a:p1:t1","index":1,"fileId":"a","fileName":"long.png","page":1,
                         "region":[0.0,1360/3000,1.0,2960/3000],"title":None}


def test_overlap_claim_belongs_to_the_tile_whose_center_is_nearest():
    # Tiles: t0 0~1600, t1 1360~2960, t2 2720~3000. 1360~1600 is shared by t0 and t1.
    manifest=tile_manifest(fixed_tiles())
    def at(y0,y1,tile_id=None):
        return {"fileId":"a","page":1,"tileId":tile_id,"bbox":[0,y0/3000,1,y1/3000]}
    assert assign_tile(at(100,140),manifest)=="a:p1:t0"
    # OCR read this line in t1 but it sits closer to t0's center; one owner, not two.
    assert assign_tile(at(1400,1440,"a:p1:t1"),manifest)=="a:p1:t0"
    assert assign_tile(at(1560,1590,"a:p1:t0"),manifest)=="a:p1:t1"
    assert assign_tile(at(2900,2950),manifest)=="a:p1:t2"


def test_region_only_and_unmatched_locations():
    manifest=tile_manifest(fixed_tiles())
    region_only={"fileId":"a","page":1,"tileId":"a:p1:t2","bbox":None,"region":manifest[2]["region"]}
    assert assign_tile(region_only,manifest)=="a:p1:t2"
    # PDF native text has a page box and no tile.
    assert assign_tile({"fileId":"a","page":1,"bbox":[0,0.5,1,0.51]},manifest)=="a:p1:t1"
    assert assign_tile({"fileId":"b","page":1,"bbox":[0,0.5,1,0.51]},manifest) is None
    assert assign_tile({"fileId":"a","page":1,"tileId":"gone","bbox":None},manifest) is None
    assert assign_tile(None,manifest) is None


async def test_missing_provider_does_not_invent_findings_or_sources(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    graph=build_graph([],[],Provider(),InMemorySaver())
    config={"configurable":{"thread_id":"tenant:synthetic:1"}}
    output=await graph.ainvoke({"reviewId":"r","tenantId":"t","versionNo":1,"product":{}},config)
    result=output["result"]
    assert result["findings"]==[] and result["visualFindings"]==[]
    assert result["pipelineStatus"]=={"text":"not_configured","visual":"not_configured"}
    assert result["legalStatus"]=="not_evaluated"
    assert result["standardsSnapshot"]==[]
    assert result["violations"]==[]
    assert output["__interrupt__"]
    resumed=await graph.ainvoke(Command(resume="approved"),config)
    assert resumed["reportReady"] is True
    assert resumed["decision"]=="approved"
    assert "runToken" not in resumed


async def test_both_paths_have_independent_results_and_checkpoint_contains_no_pixels(monkeypatch):
    from capstone_agent.review_models import TextAssessment,VisualAssessment
    import capstone_agent.review_engine as engine
    from capstone_agent.review_engine import RewriteDraft,RewriteVerification
    async def ocr(native,tiles,provider):
        return [{"id":"t1","text":"7일 주름 개선","location":{"kind":"image","fileId":"a","page":1,"bbox":None}}],[]
    monkeypatch.setattr(engine,"extract_text",ocr)
    class FakeProvider:
        client=True
        async def parse(self,prompt,schema,**kwargs):
            if schema is TextAssessment:
                return schema(findings=[{"phrase":"없는 원문","reason":"합성","claimType":"wrinkle","rewrites":[]},
                    {"phrase":"주름 개선","reason":"기간에 맞는 자료 확인","claimType":"wrinkle","rewrites":[]}])
            if schema is RewriteDraft:
                return RewriteDraft(text="자료 확인 중인 주름 개선 효과")
            if schema is RewriteVerification:
                return RewriteVerification(passed=True,reasoning="합성 재검증 통과")
            return VisualAssessment(sectionTitle="사용 전후 비교",findings=[{"observation":"합성 비교 구성","concern":"촬영 조건 확인","comparisonLimitations":[],"needsAdjacentContext":False}],unreadable=False)
    tile=tile_image(Image.new("RGB",(64,64)),"a","synthetic.png")[0]
    graph=build_graph([], [tile],FakeProvider(),InMemorySaver())
    output=await graph.ainvoke({"reviewId":"r","tenantId":"t","versionNo":1,"product":{}},
                              {"configurable":{"thread_id":"t:r:1"}})
    assert len(output["result"]["findings"])==1
    assert len(output["result"]["visualFindings"])==1
    assert [(t["id"],t["title"]) for t in output["result"]["tiles"]]==[(tile.id,"사용 전후 비교")]
    assert output["result"]["visualFindings"][0]["tileId"]==tile.id
    assert output["result"]["findings"][0]["tileId"] is None  # synthetic OCR block has no box or tile
    assert "data:image" not in str(output)
    assert output["result"]["findings"][0]["basis"]==[]
    violations=output["result"]["violations"]
    assert len(violations)==1
    assert violations[0]["text_excerpt"]=="주름 개선"
    assert violations[0]["rewrite_suggestion"]=="자료 확인 중인 주름 개선 효과"
    assert violations[0]["rewrite_attempt_count"]==1
    assert violations[0]["rewrites"]==[{"attempt_no":1,"text":"자료 확인 중인 주름 개선 효과",
        "requested_by_human":False,"retry_note":None,"passed":True,"reasoning":"합성 재검증 통과"}]
    assert violations[0]["final_resolution"] is None and violations[0]["final_text_validated"] is False


async def test_result_findings_keep_tile_and_later_severity_updates(monkeypatch):
    # The violations node updates the finding dicts in place; the result must carry both that
    # severity and the tileId set when the finding was created.
    from capstone_agent.review_models import TextAssessment,VisualAssessment
    import capstone_agent.review_engine as engine
    tile=tile_image(Image.new("RGB",(64,64)),"a","synthetic.png")[0]
    async def ocr(native,tiles,provider):
        return [{"id":"t1","text":"7일 주름 개선","location":{"kind":"image","fileId":"a","page":1,"bbox":[0,0.1,1,0.2]}}],[]
    async def severity(provider,finding,evidence_text):
        return "high","합성 상향"
    async def no_rewrites(findings,provider):
        return []
    monkeypatch.setattr(engine,"extract_text",ocr)
    monkeypatch.setattr(engine,"assess_severity",severity)
    monkeypatch.setattr(engine,"build_violations",no_rewrites)
    class FakeProvider:
        client=True
        async def parse(self,prompt,schema,**kwargs):
            if schema is TextAssessment:
                return schema(findings=[{"phrase":"주름 개선","reason":"합성","claimType":"wrinkle","rewrites":[]}])
            return VisualAssessment(sectionTitle="합성 구간",findings=[{"observation":"합성","concern":"합성","comparisonLimitations":[],"needsAdjacentContext":False}],unreadable=False)
    graph=build_graph([], [tile],FakeProvider(),InMemorySaver())
    output=await graph.ainvoke({"reviewId":"r","tenantId":"t","versionNo":1,"product":{}},
                              {"configurable":{"thread_id":"t:r:2"}})
    finding=output["result"]["findings"][0]
    assert finding["severity"]=="high" and finding["escalation"]["reason"]=="합성 상향"
    assert finding["tileId"]==tile.id
    assert output["result"]["visualFindings"][0]["tileId"]==tile.id


async def test_rewrite_retry_limit_stops_auto_suggestions_for_manual_input(monkeypatch):
    from capstone_agent.review_models import TextAssessment,VisualAssessment
    import capstone_agent.review_engine as engine
    from capstone_agent.review_engine import RewriteDraft,RewriteVerification,REWRITE_RETRY_LIMIT
    async def ocr(native,tiles,provider):
        return [{"id":"t1","text":"평생 보습 지속","location":{"kind":"image","fileId":"a","page":1,"bbox":None}}],[]
    monkeypatch.setattr(engine,"extract_text",ocr)
    attempts=[]
    class NeverPassingProvider:
        client=True
        async def parse(self,prompt,schema,**kwargs):
            if schema is TextAssessment:
                return schema(findings=[{"phrase":"평생 보습 지속","reason":"절대적 표현","claimType":"absolute","rewrites":[]}])
            if schema is RewriteDraft:
                attempts.append(1)
                return RewriteDraft(text=f"시도 {len(attempts)}")
            if schema is RewriteVerification:
                return RewriteVerification(passed=False,reasoning="여전히 절대적 표현")
            return VisualAssessment(sectionTitle="",findings=[],unreadable=False)
    graph=build_graph([],[],NeverPassingProvider(),InMemorySaver())
    output=await graph.ainvoke({"reviewId":"r","tenantId":"t","versionNo":1,"product":{}},
                              {"configurable":{"thread_id":"t:r:2"}})
    violation=output["result"]["violations"][0]
    assert violation["rewrite_attempt_count"]==REWRITE_RETRY_LIMIT==3
    assert len(violation["rewrites"])==REWRITE_RETRY_LIMIT
    assert violation["final_resolution"] is None  # 사람의 rewritten_manual 입력이 필요하다


async def test_verify_rewrite_checks_source_agnostically_and_does_not_rewrite():
    from capstone_agent.review_engine import verify_rewrite
    from capstone_agent.review_models import RewriteVerification
    seen = {}
    class FakeProvider:
        async def parse(self, prompt, schema, **kwargs):
            seen["schema"], seen["text"] = schema, kwargs["text"]
            return RewriteVerification(passed=False, reasoning="여전히 절대적 표현")
    result = await verify_rewrite(FakeProvider(), "평생 지속되는 보습", "절대적 표현")
    assert seen["schema"] is RewriteVerification
    assert "평생 지속되는 보습" in seen["text"] and "절대적 표현" in seen["text"]
    assert result.passed is False and result.reasoning


def test_patent_error_envelope_is_not_no_results():
    from capstone_business_mcp.review_tools import parse_patents
    result=parse_patents("<response><header><successYN>N</successYN><resultCode>30</resultCode></header></response>","10-1234567","synthetic")
    assert result["status"]=="error"


async def test_public_api_configuration_is_not_a_product_absence(monkeypatch):
    from capstone_business_mcp.review_tools import functional_report
    monkeypatch.delenv("MFDS_SERVICE_KEY",raising=False)
    result=await functional_report("synthetic","합성 제품")
    assert result["status"]=="error" and result["error"]=="API_NOT_CONFIGURED"


def test_source_preview_preserves_original_spans_and_page_boundaries():
    from capstone_rag_mcp.preview import chunk_pages
    pages=['제1조 본문\n'+'조건과 예외를 함께 확인한다.\n'*100, '제2조 별도 페이지']
    chunks=chunk_pages(pages)
    assert all(c['body']==pages[c['page']-1][c['start']:c['end']] for c in chunks)
    first=[c for c in chunks if c['page']==1]
    assert first[0]['start']==0 and first[-1]['end']==len(pages[0])
    assert all(a['end']>b['start'] for a,b in zip(first,first[1:]))
    assert chunks[-1]['page']==2


def test_public_functional_fields_never_invent_unknown_scope():
    from capstone_business_mcp.review_tools import functional_item
    data=functional_item({'ITEM_NAME':'합성 제품','EFFECT_YN1':'Y','EFFECT_YN2':'N','EFFECT_YN3':'unknown','SPF':'50+'})
    assert data['functions']==['미백']
    assert data['spf']=='50+'
    assert data['scopeFields']['EFFECT_YN3']=='unknown'


async def test_functional_lookup_uses_official_operation_and_catches_cancelled_items(monkeypatch):
    import httpx
    import capstone_business_mcp.review_tools as tools
    original=httpx.AsyncClient
    def handler(request):
        assert request.url.path.endswith('/FtnltCosmRptPrdlstInfoService01/getRptPrdlstInq01')
        assert request.url.params['cosmetic_report_seq']=='123'
        return httpx.Response(200,json={'header':{'resultCode':'00'},'body':{'totalCount':1,'items':[
            {'ITEM_NAME':'합성 제품','COSMETIC_REPORT_SEQ':'123','CANCEL_APPROVAL_YN':'Y'}]}})
    monkeypatch.setenv('MFDS_SERVICE_KEY','synthetic-test-key')
    monkeypatch.delenv('MFDS_REPORT_URL',raising=False)
    monkeypatch.setattr(tools.httpx,'AsyncClient',lambda **kw:original(transport=httpx.MockTransport(handler),**kw))
    result=await tools.functional_report('123','합성 제품')
    assert result['status']=='cancelled' and result['error']=='REPORTED_ITEM_CANCELLED'


async def test_functional_lookup_does_not_treat_truncated_pages_as_product_absence(monkeypatch):
    import httpx
    import capstone_business_mcp.review_tools as tools
    original=httpx.AsyncClient
    pages=[]
    def handler(request):
        pages.append(request.url.params['pageNo'])
        return httpx.Response(200,json={'header':{'resultCode':'00'},'body':{'totalCount':1000,'items':[
            {'ITEM_NAME':'다른 제품','COSMETIC_REPORT_SEQ':'987'}]}})
    monkeypatch.setenv('MFDS_SERVICE_KEY','synthetic-test-key')
    monkeypatch.delenv('MFDS_REPORT_URL',raising=False)
    monkeypatch.setattr(tools.httpx,'AsyncClient',lambda **kw:original(transport=httpx.MockTransport(handler),**kw))
    result=await tools.functional_report('123','합성 제품')
    assert pages==['1','2','3']
    assert result['status']=='ambiguous'


async def test_human_rewrite_prompt_carries_previous_suggestions_and_note():
    from capstone_agent.review_engine import rewrite_phrase
    from capstone_agent.review_models import RewriteRequest, RewriteVerification
    seen = {}
    class FakeProvider:
        client = True
        async def parse(self, prompt, schema, **kwargs):
            if schema is RewriteVerification:
                return schema(passed=True, reasoning="문제 표현이 사라짐")
            seen["schema"], seen["text"] = schema, kwargs["text"]
            return schema(text="  눈가가 한결 매끈해 보이는 사용감  ")
    request = RewriteRequest(reviewId="r", tenantId="t", phrase="7일 주름 개선", reason="기간 자료 확인",
        previousRewrites=["첫 제안"], retryNote="더 짧게")
    assert await rewrite_phrase(FakeProvider(), request, call=no_rules) == "눈가가 한결 매끈해 보이는 사용감"
    assert "첫 제안" in seen["text"] and "더 짧게" in seen["text"] and "7일 주름 개선" in seen["text"]


async def no_rules(server, tool, arguments):
    if tool == "scan_claim_lines":
        return {"matches": []}
    return {"status": "CORPUS_NOT_INDEXED", "matches": [], "hits": []}


def rule_scan(table):
    """Fake rag.scan_claim_lines over a {word: (rule id, label)} table."""
    async def call(server, tool, arguments):
        matches = []
        for index, line in enumerate(arguments["lines"]):
            hits = [(word, *rule) for word, rule in table.items() if word in line]
            if hits:
                matches.append({"index": index, "line": line, "rules": [h[1] for h in hits],
                                "labels": [h[2] for h in hits], "matched": [h[0] for h in hits]})
        return {"matches": matches}
    return call


async def test_rewrite_guard_rejects_same_fragment_kept_and_added_claims():
    from capstone_agent.review_engine import rewrite_guard
    call = rule_scan({"%": ("substantiation", "수치, 시험 표현"), "테스트": ("substantiation", "수치, 시험 표현"),
                      "저자극": ("official_endorsement", "인증 표현")})
    assert "같은" in await rewrite_guard("20.48% 개선", "20.48%  개선", call)
    assert "잘려" in await rewrite_guard("색소 침착 7.23% 개선", "색소", call)
    assert "남아" in await rewrite_guard("보습 20.48% 개선", "보습 20% 개선 효과", call)
    assert "새로" in await rewrite_guard("순한 사용감", "저자극 사용감", call)
    assert await rewrite_guard("보습 20.48% 개선", "촉촉한 보습감", call) is None
    assert "끊겼" in await rewrite_guard("건강하게 빛나는 윤기를 선사합니다", "건강하게 빛나는 윤기", call)
    assert await rewrite_guard("건강하게 빛나는 윤기를 선사합니다", "건강해 보이는 윤기를 더합니다", call) is None


async def test_business_tools_run_without_a_model_key_instead_of_being_skipped(monkeypatch):
    """Without OPENAI_API_KEY there is no LLM to ask, but structure_evidence/verify_functional_approval/
    verify_patent are business-mcp calls, independent of the model — they must still run (NFR-16),
    not silently no-op just because the unrelated tool-selection step has no provider."""
    import capstone_agent.review_engine as engine
    from capstone_agent.review_engine import select_business_tools

    class NoKeyProvider:
        client = None  # no call_with_tools method either, so this path must never touch it

    class FakeAsset:
        role = "evidence"
        def model_dump(self): return {"id": "e1", "role": "evidence"}

    class FakeRequest:
        assets = [FakeAsset()]

    mcp_calls = []
    async def fake_mcp_call(name, arguments, request=None):
        mcp_calls.append(name)
        return {"status": "found", "text": "증빙 원문"}
    monkeypatch.setattr(engine, "mcp_call", fake_mcp_call)

    state = {"tenantId": "t", "reviewId": "r", "product": {"name": "수분크림"}, "texts": [{"text": "특허 10-1234567"}],
             "findings": [{"claimType": "patent"}]}
    evidence, public, patents, warnings, trace = await select_business_tools(NoKeyProvider(), state, FakeRequest())

    assert set(mcp_calls) == {"structure_evidence", "verify_functional_approval", "verify_patent"}
    assert evidence["text"] == "증빙 원문"
    assert public == {"status": "found", "text": "증빙 원문"}


async def test_business_tool_loop_follows_llm_choice_and_guardrail_caps_repeats(monkeypatch):
    """select_business_tools: the LLM picks which available tool to call (never the one it never asks
    for, even though it is available), and a tool it keeps re-requesting stops executing once
    MAX_SAME_TOOL_CALLS is hit."""
    import capstone_agent.review_engine as engine
    from capstone_agent.review_engine import select_business_tools, MAX_SAME_TOOL_CALLS

    class FakeToolCall:
        def __init__(self, name, call_id):
            self.id = call_id
            self.function = type("F", (), {"name": name})()
        def model_dump(self):
            return {"id": self.id, "type": "function", "function": {"name": self.function.name, "arguments": "{}"}}

    class FakeMessage:
        content = ""
        def __init__(self, names):
            self.tool_calls = [FakeToolCall(n, f"{n}-{i}") for i, n in enumerate(names)] if names else None

    class FakeToolProvider:
        client = True
        def __init__(self, turns):
            self.turns = list(turns)
            self.calls = 0
        async def call_with_tools(self, messages, tools):
            names = self.turns[self.calls] if self.calls < len(self.turns) else []
            self.calls += 1
            return FakeMessage(names)

    class FakeAsset:
        role = "evidence"
        def model_dump(self): return {"id": "e1", "role": "evidence"}

    class FakeRequest:
        assets = [FakeAsset()]

    mcp_calls = []
    async def fake_mcp_call(name, arguments, request=None):
        mcp_calls.append(name)
        return {"status": "found"}
    monkeypatch.setattr(engine, "mcp_call", fake_mcp_call)

    # Model repeatedly re-requests verify_functional_approval and never once asks for
    # structure_evidence, even though both are available (evidence file submitted, product named).
    provider = FakeToolProvider([["verify_functional_approval"]] * (MAX_SAME_TOOL_CALLS + 2) + [[]])
    state = {"product": {"name": "수분크림"}, "texts": [], "findings": []}
    evidence, public, patents, warnings, trace = await select_business_tools(provider, state, FakeRequest())

    assert mcp_calls == ["verify_functional_approval"] * MAX_SAME_TOOL_CALLS  # guardrail stopped the rest
    assert "structure_evidence" not in mcp_calls  # never chosen, so never executed despite being available
    assert public == {"status": "found"}
    assert {"tool": "structure_evidence", "called": False} in trace
    assert trace.count({"tool": "verify_functional_approval", "called": True}) == MAX_SAME_TOOL_CALLS


async def test_unverified_rewrite_is_not_stored_as_suggestion():
    from capstone_agent.review_engine import build_violations
    from capstone_agent.review_models import RewriteVerification
    class FakeProvider:
        client = True
        async def parse(self, prompt, schema, **kwargs):
            if schema is RewriteVerification:
                return schema(passed=False, reasoning="새 효능을 보장함")
            return schema(text="피부 재생 완벽 케어")
    finding = {"id": "f", "phrase": "7일 주름 개선", "reason": "기간 자료", "severity": "low",
               "rewrites": ["7일 주름 개선"], "basis": []}
    [violation] = await build_violations([finding], FakeProvider(), no_rules)
    assert violation["rewrite_suggestion"] is None and not violation["verified"]
    assert violation["rewrites"][0]["reasoning"] == "원문과 같은 문구입니다."  # first candidate fails the guard
    assert violation["rewrite_note"] == "새 효능을 보장함"


def blank_pdf(pages):
    import pypdfium2 as pdfium
    document = pdfium.PdfDocument.new()
    for _ in range(pages):
        document.new_page(595, 842)
    output = BytesIO()
    document.save(output)
    return output.getvalue()


def test_evidence_reader_takes_long_reports_and_names_the_failure():
    from capstone_shared.documents import prepare_evidence
    blocks, scans, pages = prepare_evidence(blank_pdf(30), {"id": "e", "name": "report.pdf"})
    assert pages == 30 and blocks == [] and len(scans) == 30  # no text layer: every page goes to OCR
    import pytest
    with pytest.raises(ValueError, match="EVIDENCE_PAGE_LIMIT"):
        prepare_evidence(blank_pdf(81), {"id": "e", "name": "huge.pdf"})


def test_substantiation_uses_the_claim_match_and_flags_unread_evidence():
    from capstone_agent.checks import substantiation_check
    finding = {"id": "f", "phrase": "2주 안에 모공 개선"}
    matched = {"verdict": "different_test", "documentName": "인체실험 보고서.pdf", "needed": "모공 개선 인체적용시험 보고서",
               "reason": "제출 문서는 피부 일차자극 시험이라 모공 변화를 측정하지 않았습니다."}
    result = substantiation_check(finding, {"documents": [{"id": "e"}]}, 1, matched)
    assert result["status"] == "evidence_mismatch" and result["attention"]
    assert "인체실험 보고서.pdf" in result["summary"] and "모공 개선 인체적용시험 보고서" in result["summary"]
    supported = substantiation_check(finding, {"documents": [{"id": "e"}]}, 1, {**matched, "verdict": "supports", "needed": ""})
    assert supported["status"] == "evidence_supports" and not supported["attention"]
    unread = substantiation_check(finding, {"documents": [], "failures": [{"code": "EVIDENCE_PAGE_LIMIT"}]}, 1)
    assert unread["status"] == "evidence_unread" and "EVIDENCE_PAGE_LIMIT" in unread["summary"]
    not_read = substantiation_check(finding, {"status": "not_submitted"}, 1)
    assert not_read["status"] == "evidence_submitted"  # the tool loop did not read it; that is not a read failure


async def test_claim_match_keeps_only_known_claims_and_passes_ad_conditions():
    from capstone_agent.evidence import ClaimMatches, match
    seen = {}
    class FakeProvider:
        async def parse(self, prompt, schema, **kwargs):
            seen["text"] = kwargs["text"]
            return ClaimMatches(items=[
                {"findingId": "f", "documentId": "e", "verdict": "different_test", "reason": "자극 시험", "needed": "모공 시험"},
                {"findingId": "ghost", "documentId": None, "verdict": "not_covered", "reason": "-", "needed": "-"}])
    findings = [{"id": "f", "phrase": "2주 안에 개선", "requiredChecks": ["substantiation"]},
                {"id": "g", "phrase": "촉촉함", "requiredChecks": []}]
    summaries = [{"documentId": "e", "name": "r.pdf", "kind": "safety_test", "title": "t", "purpose": "자극", "measures": ["피부 일차자극"],
                  "period": "2025.01.07~01.09", "subjects": "30명", "institution": "프로브엠", "result": "저자극"}]
    result = await match(FakeProvider(), findings, summaries, ["*연구 기간 : 2024.12.30 ~ 2025.01.28 | 20명 대상 | 2주 후 비교", "촉촉한 크림"])
    assert set(result) == {"f"} and result["f"]["documentName"] == "r.pdf"
    assert "20명 대상" in seen["text"] and "촉촉한 크림" not in seen["text"]


def test_ocr_typos_in_an_institution_name_are_not_a_mismatch():
    from capstone_agent.evidence import same_institution
    assert same_institution("*연구기관 : (주)프로봄 피부임상연구센터", "㈜프로브엠 피부임상연구센터")
    assert same_institution("*연구기관: (주)프로렙 피부임상연구센터", "㈜프로브엠 피부임상연구센터")
    assert not same_institution("*연구기관: 한국피부과학연구원", "㈜프로브엠 피부임상연구센터")


def _api_error(cls, status):
    import httpx
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    return cls("synthetic", response=response, body=None)


async def test_provider_tries_fallback_model_only_for_retryable_failures(monkeypatch):
    import openai
    from capstone_agent.review_engine import Provider
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("TEXT_MODEL", "primary-model")
    monkeypatch.setenv("FALLBACK_TEXT_MODEL", "fallback-model")
    monkeypatch.setenv("MODEL_MAX_RETRIES", "2")
    provider = Provider()
    assert provider.client.max_retries == 2
    calls = []
    class Responses:
        def __init__(self, error):
            self.error = error
        async def parse(self, *, model, **kwargs):
            calls.append(model)
            if model == "primary-model":
                raise self.error
            class Done:
                output_parsed = OcrCorrection(lines=["대체 모델 결과"])
            return Done()
    class FakeClient:
        def __init__(self, error):
            self.responses = Responses(error)
    provider.client = FakeClient(_api_error(openai.RateLimitError, 429))
    result = await provider.parse("합성", OcrCorrection, text="합성")
    assert result.lines == ["대체 모델 결과"] and calls == ["primary-model", "fallback-model"]

    calls.clear()
    provider.client = FakeClient(_api_error(openai.BadRequestError, 400))
    try:
        await provider.parse("합성", OcrCorrection, text="합성")
    except openai.BadRequestError:
        pass
    else:
        raise AssertionError("a bad request must not be retried on another model")
    assert calls == ["primary-model"]

    calls.clear()
    monkeypatch.setenv("FALLBACK_TEXT_MODEL", "")
    provider.client = FakeClient(_api_error(openai.InternalServerError, 503))
    try:
        await provider.parse("합성", OcrCorrection, text="합성")
    except openai.InternalServerError:
        pass
    else:
        raise AssertionError("without a fallback model the provider error must surface")
    assert calls == ["primary-model"]
