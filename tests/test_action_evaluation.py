import json
from pathlib import Path

from action_evaluation import case_action, case_texts, parse_condition, render, summarize, to_csv, validate

ROOT = Path(__file__).resolve().parents[1]


def test_2026_10_22_set_is_scorable():
    data = json.loads((ROOT / "evals" / "2026-10-22" / "harness_cases.json").read_text(encoding="utf-8"))
    assert validate(data) == []


def test_condition_holds_only_product_name_and_evidence_state():
    assert parse_condition("제품명: 투집 포어본드. 증빙: 미제출.") == {"name": "투집 포어본드"}
    assert parse_condition("증빙: PDF 첨부.") == {}
    bad = {"cases": [{"id": "X", "kind": "답 가능", "text": "모공 커버", "expected": "참고", "evidenceFiles": [],
                      "condition": "사용 전후와 블러처리라는 프라이머 광고 맥락."},
                     {"id": "Y", "kind": "답 가능", "text": "탄력", "expected": "증빙 필요", "evidenceFiles": [], "condition": "증빙: PDF 첨부."},
                     {"id": "Z", "kind": "어려움", "text": "탄력", "expected": "증빙 필요", "evidenceFiles": [], "condition": "증빙: 미제출."}]}
    problems = validate(bad)
    assert any(p.startswith("X: condition must be") for p in problems)
    assert any(p.startswith("Y: condition evidence state") for p in problems)
    assert any(p.startswith("Z: kind must be") for p in problems)
    texts = case_texts({"id": "E02", "text": "*처지고 늘어진 모공\n2주 안에 개선\n*일시적 모공 개선 평가 기준"})
    assert [t["text"] for t in texts] == ["*처지고 늘어진 모공", "2주 안에 개선", "*일시적 모공 개선 평가 기준"]


def test_case_action_is_the_strongest_card_or_no_card():
    assert case_action(["note", "evidence"]) == "증빙 필요"
    assert case_action(["hold", "revise"]) == "수정 권고"
    assert case_action([]) == "검토 항목 없음"


def test_cases_are_counted_once_by_how_many_runs_matched():
    def case(cid, expected, kind="답 가능", files=()):
        return {"id": cid, "kind": kind, "text": "문구", "condition": "증빙: PDF 첨부." if files else "증빙: 미제출.",
                "expected": expected, "reason": "근거", "evidenceFiles": list(files)}
    cases = [case("A", "수정 권고"), case("B", "참고", "실패 유도", ["fixtures/report.pdf"]),
             case("C", "증빙 필요"), case("D", "검토 항목 없음")]

    def outcome(cid, actual, error=None):
        cards = [{"phrase": "문구", "action": "evidence"}, {"phrase": "둘째", "action": "note"}] if actual == "증빙 필요" else []
        return {"id": cid, "actual": actual, "error": error, "cards": cards}
    runs = [[outcome("A", "수정 권고"), outcome("B", "증빙 필요"), outcome("C", "증빙 필요"), outcome("D", None, "RuntimeError")],
            [outcome("A", "수정 권고"), outcome("B", "증빙 필요"), outcome("C", "참고"), outcome("D", None, "RuntimeError")],
            [outcome("A", "수정 권고"), outcome("B", "증빙 필요"), outcome("C", "증빙 필요"), outcome("D", None, "RuntimeError")]]
    summary = summarize(cases, runs)
    assert (summary["all"], summary["some"], summary["none"], summary["unscored"]) == (1, 1, 1, 1)
    config = {"checkedAt": "2026-10-22T00:00:00+00:00", "casesVersion": "2026-10-22.2", "textModel": "m", "promptHash": "p",
              "promptDate": "2026-10-21", "rulesVersion": "r", "reportPath": "reports/eval/x.json", "evidenceWarnings": {}}
    report = {"config": config, "summary": summary, "runs": runs}
    text = render(report, cases)
    assert "**테스트 환경** 모델: m, 프롬프트 버전: 2026.10.21.(text_review p), 규칙 버전: r, 데이터 버전: 2026-10-22.2, 실행일: 2026.10.22." in text
    assert "| 2 | 실패 유도 | B: 문구 (증빙 report.pdf) | 참고. 근거 | 3회 모두 증빙 필요<br>카드: \"문구\" 증빙 필요<br>\"둘째\" 참고 | 미충족 |  |" in text
    assert "| 일부 충족 (2/3) |" in text and "| 실행 오류 | RuntimeError; RuntimeError; RuntimeError |" in text
    assert "**결과: 개선 전 1 / 4 → 개선 후 - / 4 (충족률 25.0% → -, 평가셋 2026-10-22.2)**" in text
    assert "충족 1건, 일부 충족 1건(C), 미충족 1건이다. 일부 충족까지 포함하면 2 / 4(50.0%)다." in text
    better = {**report, "summary": {**summary, "all": 3}}
    assert "개선 전 1 / 4 → 개선 후 3 / 4 (충족률 25.0% → 75.0%" in render(better, cases, before=report)
    rows = to_csv(report, cases).splitlines()
    assert rows[0] == "No.,유형,점검 입력,기대 결과 (근거 포함),실제 출력 요약,충족,비고" and len(rows) == 5
