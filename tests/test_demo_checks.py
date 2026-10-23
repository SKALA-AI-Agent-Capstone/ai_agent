from capstone_agent.checks import check, patent_check, resolved_by_evidence


def card(fid, claim, *checks):
    return {"id": fid, "phrase": fid, "textId": f"t-{fid}", "claimType": claim, "checks": list(checks)}


def supports():
    out = check("substantiation", "evidence_supports", "보고서: 같은 시험입니다.")
    out["detail"]["match"] = {"documentId": "d1", "documentName": "보고서.pdf"}
    return out


def test_a_card_a_document_supports_leaves_as_resolved_with_the_document():
    kept, resolved = resolved_by_evidence([
        card("a", "clinical", supports()),
        card("b", "clinical", check("substantiation", "evidence_mismatch", "다른 시험")),
        card("c", "drug", supports()),  # a medical-effect expression is the problem itself
        card("d", "clinical", supports(), check("patent", "needs_input", "번호 없음")),
    ])
    assert [f["id"] for f in kept] == ["b", "c", "d"]
    assert resolved == [{"id": "a", "phrase": "a", "textId": "t-a", "source": "evidence_supports",
                         "reason": "보고서: 같은 시험입니다.", "documentId": "d1", "documentName": "보고서.pdf"}]


def test_the_patent_check_says_when_the_printed_patent_has_lapsed():
    lapsed = [{"number": "10-0440357", "status": "expired", "data": {"registerDate": "20040705"}}]
    out = patent_check(["10-0440357"], lapsed)
    assert out["status"] == "patent_expired" and out["attention"] and "2004년 등록" in out["summary"]
    assert patent_check(["10-1"], [{"number": "10-1", "status": "registered"}])["status"] == "patent_registered"
    assert patent_check([], None)["status"] == "needs_input"
    assert patent_check(["10-2"], [])["status"] == "evidence_submitted"
