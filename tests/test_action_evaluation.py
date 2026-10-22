import json
from pathlib import Path

from action_evaluation import case_action, case_texts, parse_condition, render, summarize, validate

ROOT = Path(__file__).resolve().parents[1]


def test_2026_10_22_set_is_scorable():
    data = json.loads((ROOT / "evals" / "2026-10-22" / "harness_cases.json").read_text(encoding="utf-8"))
    assert validate(data) == []


def test_condition_holds_only_product_name_and_evidence_state():
    assert parse_condition("제품명: 투집 포어본드. 증빙: 미제출.") == {"name": "투집 포어본드"}
    assert parse_condition("증빙: PDF 첨부.") == {}
    bad = {"cases": [{"id": "X", "text": "모공 커버", "expected": "참고", "evidenceFiles": [],
                      "condition": "사용 전후와 블러처리라는 프라이머 광고 맥락."},
                     {"id": "Y", "text": "탄력", "expected": "증빙 필요", "evidenceFiles": [], "condition": "증빙: PDF 첨부."}]}
    problems = validate(bad)
    assert any(p.startswith("X: condition must be") for p in problems)
    assert any(p.startswith("Y: condition evidence state") for p in problems)
    texts = case_texts({"id": "E02", "text": "*처지고 늘어진 모공\n2주 안에 개선\n*일시적 모공 개선 평가 기준"})
    assert [t["text"] for t in texts] == ["*처지고 늘어진 모공", "2주 안에 개선", "*일시적 모공 개선 평가 기준"]


def test_case_action_is_the_strongest_card_or_no_card():
    assert case_action(["note", "evidence"]) == "증빙 필요"
    assert case_action(["hold", "revise"]) == "수정 권고"
    assert case_action([]) == "검토 항목 없음"


def test_cases_are_counted_once_by_how_many_runs_matched():
    cases = [{"id": "A", "expected": "수정 권고"}, {"id": "B", "expected": "참고"},
             {"id": "C", "expected": "증빙 필요"}, {"id": "D", "expected": "검토 항목 없음"}]

    def outcome(cid, actual, error=None):
        return {"id": cid, "actual": actual, "error": error, "cards": [{"phrase": "문구", "action": "evidence"}] if actual else []}
    runs = [[outcome("A", "수정 권고"), outcome("B", "증빙 필요"), outcome("C", "증빙 필요"), outcome("D", None, "RuntimeError")],
            [outcome("A", "수정 권고"), outcome("B", "증빙 필요"), outcome("C", "참고"), outcome("D", None, "RuntimeError")]]
    summary = summarize(cases, runs)
    assert (summary["all"], summary["some"], summary["none"], summary["unscored"]) == (1, 1, 1, 1)
    assert summary["scored"] == 6 and summary["matched"] == 3
    config = {"checkedAt": "2026-10-22T00:00:00+00:00", "label": "t", "runs": 2, "casesVersion": "v", "cases": 4,
              "textModel": "m", "promptHash": "p", "rulesVersion": "r", "commit": "c", "reportPath": "reports/eval/x.json"}
    text = render({"config": config, "summary": summary}, runs[-1])
    assert "| 합계 | 4 | 1 | 1 | 1 |" in text
    assert "회차 기준: 실행 6번(사례 4개 x 2회, 오류 회차 제외) 중 3번 일치(50.0%)." in text
    assert "- B: 기대 참고, 회차별 실제 증빙 필요, 증빙 필요 (0/2 일치)." in text
    assert "- D: RuntimeError" in text and "모든 회차가 실행 오류라 채점하지 못한 사례: 1개." in text
    assert "### 증빙 추출 경고\n\n- 없음" in text
    noted = render({"config": {**config, "evidenceWarnings": {"P05": ["OCR_UNAVAILABLE p.28"]}}, "summary": summary}, runs[-1])
    assert "- P05: OCR_UNAVAILABLE p.28" in noted
