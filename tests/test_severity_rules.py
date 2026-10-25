"""Claims in an ad that pictures a lapsed patent are raised to high; other files keep the model's severity."""
from capstone_agent.severity_rules import raise_for_lapsed_patents

PATENT_AD = {"fileId": "ad1"}
TEXTS = [{"id": "t1", "text": "특 허 제 0440357 호 출원번호 제 2001-0065893 호", "location": PATENT_AD},
         {"id": "t2", "text": "볼륨 개선도 92.7%", "location": PATENT_AD},
         {"id": "t3", "text": "촉촉한 크림", "location": {"fileId": "ad2"}}]


def card(i, severity, file_id="ad1"):
    return {"id": i, "phrase": i, "severity": severity, "location": {"fileId": file_id}}


def test_every_claim_in_the_file_that_shows_a_lapsed_patent_is_high_with_the_reason():
    patents = [{"number": "10-0440357", "status": "expired"}]
    out = raise_for_lapsed_patents([card("a", "medium"), card("b", "low"), card("c", "high"), card("d", "medium", "ad2")],
                                   TEXTS, patents)
    assert [f["severity"] for f in out] == ["high", "high", "high", "medium"]
    assert out[0]["escalation"]["from"] == "medium" and "10-0440357" in out[0]["escalation"]["reason"]
    assert "escalation" not in out[2] and "escalation" not in out[3]       # already high, and another file


def test_a_registered_or_unknown_patent_changes_nothing():
    cards = [card("a", "medium")]
    assert raise_for_lapsed_patents(cards, TEXTS, [{"number": "10-0440357", "status": "registered"}]) == cards
    assert raise_for_lapsed_patents(cards, TEXTS, [{"number": "10-0440357", "status": "unavailable"}]) == cards
    assert raise_for_lapsed_patents(cards, TEXTS, None) == cards
    assert raise_for_lapsed_patents(cards, TEXTS, [{"number": "10-9999999", "status": "expired"}]) == cards  # not pictured
