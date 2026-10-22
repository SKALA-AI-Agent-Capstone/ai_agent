"""Harness evaluation set 2026-10-22: the reviewer's next action per case (evals/2026-10-22/harness_cases.json).

Each case is one ad copy with an expected outcome: 수정 권고, 증빙 필요, 판단 보류, 참고 or 검토 항목 없음.
The run (scripts/evaluate_actions.py) sends the copy through the review steps and the frontend's actionFor.
A case's action is the strongest action among its cards (revise > evidence > hold > note), and no card at all
is 검토 항목 없음. `condition` holds only the product name and the evidence state; the product name reaches the
model as the product the frontend sends, and `expected` and `reason` are for scoring only.
"""
import re

LABELS = {"revise": "수정 권고", "evidence": "증빙 필요", "hold": "판단 보류", "note": "참고", None: "검토 항목 없음"}
ORDER = ["revise", "evidence", "hold", "note"]
EXPECTED = list(LABELS.values())


CONDITION = re.compile(r"^(제품명: (?P<name>[^.]+)\. )?증빙: (미제출|PDF 첨부)\.$")


def parse_condition(condition: str) -> dict:
    """The product the frontend would send. The rest of `condition` is the evidence state, given by evidenceFiles."""
    match = CONDITION.match(condition or "")
    return {"name": match["name"]} if match and match["name"] else {}


def case_texts(case: dict) -> list[dict]:
    """Ad copy lines, footnotes included, in the shape extract_text returns."""
    lines = [line for line in case["text"].split("\n") if line.strip()]
    return [{"id": f"{case['id']}:{n}", "text": line, "location": {"kind": "copy"}} for n, line in enumerate(lines)]


def case_action(actions: list[str]) -> str:
    return LABELS[next((a for a in ORDER if a in actions), None)]


def validate(data: dict) -> list[str]:
    problems = []
    for case in data["cases"]:
        if case.get("expected") not in EXPECTED:
            problems.append(f"{case['id']}: unknown expected {case.get('expected')!r}")
        if not case.get("text", "").strip():
            problems.append(f"{case['id']}: empty text")
        if not CONDITION.match(case.get("condition", "")):
            problems.append(f"{case['id']}: condition must be '제품명: X. 증빙: 미제출.' or '증빙: PDF 첨부.' form")
        elif ("PDF 첨부" in case["condition"]) != bool(case.get("evidenceFiles")):
            problems.append(f"{case['id']}: condition evidence state differs from evidenceFiles")
        if case.get("fault") not in (None, "evidence_parse_error"):
            problems.append(f"{case['id']}: unsupported fault {case['fault']}")
        if case.get("fault") and not case.get("evidenceFiles"):
            problems.append(f"{case['id']}: fault needs an evidence file")
    return problems


def summarize(cases: list[dict], runs: list[list[dict]]) -> dict:
    """runs[r][i] is case i's outcome in run r: {"actual": label, "error": str|None, ...}.
    A run with an error is not scored as right or wrong. Each case is counted once by how many of its scored
    runs matched: all, some or none. The run-level count (cases x runs) is kept as a secondary figure."""
    by_label = {label: {"cases": 0, "all": 0, "some": 0, "none": 0, "unscored": 0, "scored": 0, "matched": 0}
                for label in EXPECTED}
    rows = []
    for i, case in enumerate(cases):
        outcomes = [run[i] for run in runs]
        scored = [o for o in outcomes if not o["error"]]
        matched = sum(o["actual"] == case["expected"] for o in scored)
        row = by_label[case["expected"]]
        row["cases"] += 1
        row["scored"] += len(scored)
        row["matched"] += matched
        row["unscored" if not scored else "all" if matched == len(scored) else "none" if not matched else "some"] += 1
        rows.append({"id": case["id"], "expected": case["expected"], "actual": [o["actual"] for o in outcomes],
                     "matched": matched, "scored": len(scored), "errors": [o["error"] for o in outcomes if o["error"]]})
    totals = {key: sum(r[key] for r in by_label.values()) for key in ("all", "some", "none", "unscored", "scored", "matched")}
    return {**totals, "caseCount": len(cases), "runs": len(runs), "byExpected": by_label, "cases": rows}


def percent(part: int, whole: int) -> str:
    return f"{part / whole * 100:.1f}%" if whole else "-"


def render(report: dict, outcomes: list[dict]) -> str:
    """Section appended to the 2026-10-22 results in backend/docs/13-harness-evaluation.md.
    outcomes is the last run, for the details."""
    config, summary = report["config"], report["summary"]
    lines = [f"## {config['checkedAt'][:10]} 실행 ({config['label']})", "",
             f"- 실행 {config['checkedAt']}, 반복 {config['runs']}회, 평가셋 {config['casesVersion']}({config['cases']}개)",
             f"- 모델 {config['textModel']}, text_review 프롬프트 {config['promptHash']}, 규칙 {config['rulesVersion']}, 커밋 {config['commit']}",
             "- 실행 범위: 문구 추출, 규칙 sweep과 근거 조항, 증빙 추출(fixtures PDF), 증빙 요약과 대조, 교차 확인, 심각도, 프론트 actionFor. "
             "광고 이미지 OCR, 업무 도구 선택 루프, 수정안 생성은 제외했다.",
             "- 사례의 실제 결과는 카드 중 가장 강한 행동(수정 권고, 증빙 필요, 판단 보류, 참고 순)이고 카드가 없으면 검토 항목 없음이다. "
             "실행 오류가 난 사례는 맞음과 틀림에서 뺀다.",
             "", "| 기대한 행동 | 사례 | 모든 회차 일치 | 일부 회차 일치 | 한 번도 안 맞음 |", "| --- | ---: | ---: | ---: | ---: |"]
    for label, row in summary["byExpected"].items():
        lines.append(f"| {label} | {row['cases']} | {row['all']} | {row['some']} | {row['none']} |")
    lines.append(f"| 합계 | {summary['caseCount']} | {summary['all']} | {summary['some']} | {summary['none']} |")
    lines += ["", f"회차 기준: 실행 {summary['scored']}번(사례 {summary['caseCount']}개 x {summary['runs']}회, 오류 회차 제외) 중 "
                  f"{summary['matched']}번 일치({percent(summary['matched'], summary['scored'])})."]
    if summary["unscored"]:
        lines.append(f"모든 회차가 실행 오류라 채점하지 못한 사례: {summary['unscored']}개.")
    lines += ["", "### 기대와 다른 사례", ""]
    detail = {o["id"]: o for o in outcomes}
    mismatched = [r for r in summary["cases"] if r["matched"] < r["scored"]]
    for row in mismatched:
        cards = "; ".join(f"\"{c['phrase']}\" {LABELS[c['action']]}" for c in detail[row["id"]]["cards"]) or "카드 없음"
        lines.append(f"- {row['id']}: 기대 {row['expected']}, 회차별 실제 {', '.join(map(str, row['actual']))} "
                     f"({row['matched']}/{row['scored']} 일치). 마지막 회차의 카드: {cards}")
    if not mismatched:
        lines.append("- 없음")
    errors = [(r["id"], e) for r in summary["cases"] for e in r["errors"]]
    lines += ["", "### 실행 오류", ""] + ([f"- {cid}: {e}" for cid, e in errors] or ["- 없음"])
    # Pages the extraction could not read. The case still runs on the rest of the document.
    extraction = config.get("evidenceWarnings") or {}
    lines += ["", "### 증빙 추출 경고", ""] + ([f"- {cid}: {', '.join(codes)}" for cid, codes in extraction.items()] or ["- 없음"])
    lines += ["", f"원본 결과: ai_agent의 `{config['reportPath']}` (Git 제외)", ""]
    return "\n".join(lines)
