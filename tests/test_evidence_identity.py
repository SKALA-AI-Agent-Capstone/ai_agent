"""Evidence identity: issuer, code, version, and the pages of each product. Text is synthetic; the layouts follow
the two real clinical-test reports checked by hand (a code split by spaces on the cover, a revision table, a
report covering several products)."""
from capstone_shared import evidence_identity as eid
from capstone_shared.evidence_parse import describe, parse_evidence
from capstone_shared.reading import PageText, ReadResult

KSRC_PAGES = [
    "인체적용시험 보고서\nVersion No. 1.0\n‘알파 크림 외 1’의 인체적용시험\nHBSE-MG E-24032\n2024년 04월",
    "2 / 27 페이지\n제 출 문\nKSRC 한국피부임상연구센터는 의뢰기관에서 요청한 시험을 수행하고 보고합니다.",
    "3 / 27 페이지\n연구용역 정보\n연 구 코 드 HBSE-MGE-24032\n연구 개시일 2024. 03. 06\n기 관 명 KSRC 한국피부임상연구센터\n대상자 코드 HBSE-24032-19",
    "6 / 27 페이지\n연구 코드 HBSE-MGE-24032 연구 기간 2024. 03. 19 ~ 04. 03\n#1: 알파 크림\n#2: 베타 마스크 프로\n"
    "1. 즉각적인 속 수분량 평가\n분석 결과 사용 직후 ‘알파 크림’이 10.10% 변화율로 유의하게 증가되었다(p<0.05).\n"
    "2. 즉각적인 투명도(맑기) 평가\n분석 결과 사용 직후 ’베타 마스크 프로’가 9.71% 변화율로 유의하게 증가되었다(p<0.05).",
    "9 / 27 페이지\n3. 시험 제품 알파 크림 전성분 정제수, 글리세린\n베타 마스크 프로 전성분 정제수, 프로판다이올",
    "17 / 27 페이지\n7-2. 즉각적인 투명도(맑기) 평가\n베타 마스크 프로 사용 직후 530.21 (9.71%)",
]
PROBEM_PAGES = [
    "PM-25J17-SA017\n인체적용시험 결과보고서 ㈜프로브엠 피부임상연구센터\n“투집 포어본드”의 피부 일차자극 평가",
    "PM-25J17-SA017\n인체적용시험 결과보고서 제·개정 이력\n제·개정 관리번호 승인 일자 주요 내용\nPM-25J17-SA017 2025. 01. 17 최초 발행\n㈜프로브엠 피부임상연구센터",
    "PM-25J17-SA017\n시료번호 PM-25SA-018 연구기간 2025.01.07~2025.01.09 ㈜프로브엠",
]


def test_known_issuer_is_found_by_its_name_on_several_pages():
    assert eid.detect_issuer(KSRC_PAGES).key == "ksrc"
    assert eid.detect_issuer(PROBEM_PAGES).key == "probem"
    assert eid.detect_issuer(["아무 기관 이름도 없는 문서입니다. " * 5] * 4) is None


def test_issuer_format_takes_the_exact_code_even_when_a_neighbouring_number_is_glued_on():
    # The cover prints the code split by a space and the year right after it; squashed, "HBSE-MGE-24032" + "2024".
    codes = eid.extract_codes(KSRC_PAGES, eid.detect_issuer(KSRC_PAGES))
    assert codes == [{"label": "연구 코드", "value": "HBSE-MGE-24032", "pages": 3, "status": "confirmed"}]
    # A subject code such as HBSE-24032-19 is not a study code.
    assert all("-19" not in code["value"] for code in codes)


def test_issuer_with_two_labelled_codes_keeps_both_under_their_labels():
    codes = eid.extract_codes(PROBEM_PAGES, eid.detect_issuer(PROBEM_PAGES))
    assert {(c["label"], c["value"]) for c in codes} == {("연구번호", "PM-25J17-SA017"), ("시료번호", "PM-25SA-018")}


def test_a_study_of_another_test_on_the_same_sample_keeps_its_own_study_number():
    # 모공부위 리프팅 시험: 자극 시험과 시료번호는 같고 연구번호의 시험 기호(LI)만 다르다.
    lifting = [page.replace("PM-25J17-SA017", "PM-25J24-LI003").replace("피부 일차자극", "안면(모공부위) 리프팅 개선")
               for page in PROBEM_PAGES]
    codes = eid.extract_codes(lifting, eid.detect_issuer(lifting))
    assert {(c["label"], c["value"]) for c in codes} == {("연구번호", "PM-25J24-LI003"), ("시료번호", "PM-25SA-018")}
    assert all(c["value"] != "PM-25SA-018" for c in codes if c["label"] == "연구번호")


def test_a_code_seen_on_one_page_is_not_confirmed_and_two_candidates_are_ambiguous():
    pages = ["한국피부임상연구센터 HBSE-MGE-24032"] + ["한국피부임상연구센터 본문"] * 3
    assert eid.extract_codes(pages, eid.detect_issuer(pages))[0]["status"] == "single_page"
    pages = ["한국피부임상연구센터 HBSE-MGE-24032", "한국피부임상연구센터 HBSE-MGE-24033"]
    code = eid.extract_codes(pages, eid.detect_issuer(pages))[0]
    assert code["status"] == "ambiguous" and code["alternatives"] == ["HBSE-MGE-24033"]


def test_unregistered_issuer_gets_labelled_codes_that_still_need_a_person():
    pages = [f"{n} / 9 페이지\n연구번호: AB-2025-0042\n본문 {n}" for n in range(1, 5)]
    assert eid.detect_issuer(pages) is None
    assert eid.extract_codes(pages, None) == [{"label": "연구번호", "value": "AB-2025-0042", "pages": 4, "status": "probable"}]
    assert eid.extract_codes(pages[:1], None)[0]["status"] == "unconfirmed"
    assert eid.extract_codes(["본문만 있는 쪽"], None) == []


def test_version_comes_from_the_cover_or_the_revision_table():
    assert eid.extract_version(KSRC_PAGES) == ("Version 1.0 · 2024-04", "2024-04")
    assert eid.extract_version(PROBEM_PAGES) == ("개정 이력 1건 · 2025-01-17", "2025-01-17")
    reissued = [PROBEM_PAGES[0], PROBEM_PAGES[1].replace(
        "PM-25J17-SA017 2025. 01. 17 최초 발행", "PM-25J17-SA017 2025. 01. 17 최초 발행\nPM-25J17-SA017 2025. 03. 02 시료 정보 정정")]
    # The code is the same; the revision table is what shows it is a new edition.
    assert eid.extract_version(reissued) == ("개정 이력 2건 · 2025-03-02", "2025-03-02")
    assert eid.extract_version(["아무것도 없는 쪽"]) == (None, None)


def test_products_listed_in_the_report_get_their_own_pages_and_results():
    names = eid.product_names(KSRC_PAGES)
    assert names == ["알파 크림", "베타 마스크 프로"]
    pages = eid.product_pages(KSRC_PAGES, names)
    assert pages == [[1, 4], [4, 5, 6]] or pages[1] == [4, 5, 6]
    results = eid.extract_results(KSRC_PAGES, names)
    assert [(r["product"], r["item"], r["changePct"], r["page"], r["significant"]) for r in results] == [
        ("알파 크림", "즉각적인 속 수분량 평가", 10.10, 4, True),
        ("베타 마스크 프로", "즉각적인 투명도(맑기) 평가", 9.71, 4, True)]


def test_single_product_report_uses_the_quoted_name_on_the_cover():
    assert eid.product_names(PROBEM_PAGES) == ["투집 포어본드"]
    assert eid.product_names(["표지에 따옴표가 없는 문서"]) == []


def test_registered_product_matches_despite_spacing_typos_and_short_names():
    names = ["디퍼앤디퍼 스카워터링 크라이오 크림 마스크", "디퍼앤디퍼 핑크 리프팅 업 토닝 크림 마스크", "디퍼앤디퍼 세라루로닉 워터 코어 크림"]
    assert eid.match_product("디퍼앤디퍼 핑크 리프팅 업토닝 크림 마스크", names)[0] == 1     # spacing
    assert eid.match_product("디퍼앤디퍼 스카이워터링 크라이오 크림 마스크", names)[0] == 0   # the report misspells 스카이
    assert eid.match_product("핑크 리프팅 업 토닝", names)[0] == 1                           # a shorter name
    assert eid.match_product("투집 포어본드", names) is None                                  # another product


def test_pages_are_grouped_into_ranges():
    assert eid.group_ranges([6, 7, 9, 10, 17, 22, 17]) == [[6, 7], [9, 10], [17, 17], [22, 22]]


def read_result(pages: list[str]) -> ReadResult:
    return ReadResult([PageText(n, "text", "text_layer", text) for n, text in enumerate(pages, 1)], len(pages))


def test_describe_summarises_one_file_for_the_registered_product():
    result = describe(read_result(KSRC_PAGES), "베타 마스크 프로")
    assert result["issuer"]["key"] == "ksrc" and result["version"] == "Version 1.0 · 2024-04"
    assert result["codes"][0]["value"] == "HBSE-MGE-24032"
    match = result["productMatch"]
    assert match["name"] == "베타 마스크 프로" and match["score"] == 1.0 and match["results"][0]["changePct"] == 9.71
    assert result["warnings"] == []
    other = describe(read_result(KSRC_PAGES), "전혀 다른 제품")
    assert other["productMatch"] is None and "PRODUCT_NOT_IN_REPORT" in other["warnings"]


def test_describe_flags_a_report_whose_code_cannot_be_trusted():
    result = describe(read_result(["본문뿐인 문서 " * 10] * 3), None)
    assert result["issuer"] is None and result["codes"] == [] and "CODE_NEEDS_CONFIRMATION" in result["warnings"]


def test_parse_evidence_reads_a_pdf_end_to_end():
    from test_reading import REPORT, text_pdf

    result = parse_evidence(text_pdf(REPORT), "아무 제품", max_ocr_pages=0)
    assert result["pageCount"] == 1 and result["pages"][0]["method"] == "text_layer"
    assert result["readerVersion"] and result["unreadPages"] == []
