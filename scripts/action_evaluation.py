"""Harness evaluation set 2026-10-22: the review result per ad copy (evals/2026-10-22/harness_cases.json).

Each case is one ad copy with an expected result: 수정 권고, 증빙 필요, 판단 보류, 참고 or 검토 항목 없음.
The run (scripts/evaluate_actions.py) sends the copy through the review steps and the frontend's actionFor.
A case's result is the strongest one among its cards (revise > evidence > hold > note), and no card at all
is 검토 항목 없음. `condition` holds only the product name and the evidence state; the product name reaches the
model as the product the frontend sends, and `expected` and `reason` are for scoring only.
"""
import re

LABELS = {"revise": "수정 권고", "evidence": "증빙 필요", "hold": "판단 보류", "note": "참고", None: "검토 항목 없음"}
ORDER = ["revise", "evidence", "hold", "note"]
EXPECTED = list(LABELS.values())
# 답 가능: a normal input. 답 불가: the review cannot decide and must say so. 실패 유도: a trap such as an
# instruction inside the ad or a report for another product.
KINDS = ("답 가능", "답 불가", "실패 유도")


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
        if case.get("kind") not in KINDS:
            problems.append(f"{case['id']}: kind must be one of {', '.join(KINDS)}")
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


def status(row: dict) -> str:
    """충족 when every scored run matched, 일부 충족 when some did, 미충족 when none did."""
    if not row["scored"]:
        return "실행 오류"
    if row["matched"] == row["scored"]:
        return "충족"
    return f"일부 충족 ({row['matched']}/{row['scored']})" if row["matched"] else "미충족"


def cell(text: str) -> str:
    return text.replace("|", "/").replace("\n", " / ")


def validation_rows(cases: list[dict], report: dict) -> list[dict]:
    """One row per case in the validation report layout. Cards are those of the last run."""
    rows = {r["id"]: r for r in report["summary"]["cases"]}
    last = {o["id"]: o for o in report["runs"][-1]}
    warnings = report["config"].get("evidenceWarnings") or {}
    out = []
    for n, case in enumerate(cases, 1):
        row = rows[case["id"]]
        product = parse_condition(case["condition"]).get("name")
        files = ", ".join(f.rsplit("/", 1)[-1] for f in case["evidenceFiles"]) or "미제출"
        actual = row["actual"]
        summary = f"3회 모두 {actual[0]}" if len(actual) == 3 and len(set(actual)) == 1 else ", ".join(map(str, actual))
        cards = last[case["id"]]["cards"]
        lines = [summary]
        if cards:
            lines += [("카드: " if i == 0 else "") + f"\"{cell(c['phrase'])}\" {LABELS[c['action']]}" for i, c in enumerate(cards)]
        else:
            lines.append("카드 없음")
        notes = []
        if case.get("fault") == "evidence_parse_error":
            notes.append("증빙 PDF 파서 오류를 주입")
        if warnings.get(case["id"]) and not case.get("fault"):
            notes.append("증빙 추출 경고: " + ", ".join(warnings[case["id"]]))
        notes += row["errors"]
        out.append({"no": n, "kind": case["kind"], "id": case["id"],
                    "input": f"{cell(case['text'])} ({'제품명 ' + product + ', ' if product else ''}증빙 {files})",
                    "expected": f"{case['expected']}. {cell(case['reason'])}",
                    "actual": lines, "status": status(row), "note": "; ".join(notes)})
    return out


def result_line(report: dict, before: dict | None = None) -> str:
    s, version = report["summary"], report["config"]["casesVersion"]
    met, total = s["all"], s["caseCount"]
    if before:
        b = before["summary"]
        return (f"결과: 개선 전 {b['all']} / {b['caseCount']} → 개선 후 {met} / {total} "
                f"(충족률 {percent(b['all'], b['caseCount'])} → {percent(met, total)}, 평가셋 {version})")
    return f"결과: 개선 전 {met} / {total} → 개선 후 - / {total} (충족률 {percent(met, total)} → -, 평가셋 {version})"


def render(report: dict, cases: list[dict], before: dict | None = None) -> str:
    """The 검증 결과 section of backend/docs/13-harness-evaluation/<set date>.md: test environment, one row per
    case and the result line. Pass the earlier report as before to fill 개선 전 and 개선 후."""
    config, s = report["config"], report["summary"]
    day = config["checkedAt"][:10].replace("-", ".") + "."
    prompt = (config.get("promptDate") or "").replace("-", ".")
    lines = [f"**테스트 환경** 모델: {config['textModel']}, 프롬프트 버전: {prompt + '.' if prompt else '-'}"
             f"(text_review {config['promptHash']}), 규칙 버전: {config['rulesVersion']}, "
             f"데이터 버전: {config['casesVersion']}, 실행일: {day}", "",
             "| No. | 유형 | 점검 입력 | 기대 결과 (근거 포함) | 실제 출력 요약 | 충족 | 비고 |",
             "| ---: | --- | --- | --- | --- | --- | --- |"]
    for r in validation_rows(cases, report):
        lines.append(f"| {r['no']} | {r['kind']} | {r['id']}: {r['input']} | {r['expected']} | "
                     f"{'<br>'.join(r['actual'])} | {r['status']} | {r['note']} |")
    partial = [r["id"] for r in s["cases"] if r["scored"] and 0 < r["matched"] < r["scored"]]
    lines += ["", f"**{result_line(report, before)}**", "",
              f"충족 {s['all']}건, 일부 충족 {s['some']}건{'(' + ', '.join(partial) + ')' if partial else ''}, 미충족 {s['none']}건이다. "
              f"일부 충족까지 포함하면 {s['all'] + s['some']} / {s['caseCount']}"
              f"({percent(s['all'] + s['some'], s['caseCount'])})다."]
    if s["unscored"]:
        lines.append(f"모든 회차가 실행 오류라 채점하지 못한 사례: {s['unscored']}건.")
    lines += ["", f"원본 결과: ai_agent의 `{config['reportPath']}` (Git 제외)", ""]
    return "\n".join(lines)


def to_csv(report: dict, cases: list[dict]) -> str:
    """The same rows as a CSV, for attaching when the table is long."""
    import csv
    import io
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["No.", "유형", "점검 입력", "기대 결과 (근거 포함)", "실제 출력 요약", "충족", "비고"])
    for r in validation_rows(cases, report):
        writer.writerow([r["no"], r["kind"], f"{r['id']}: {r['input']}", r["expected"], " / ".join(r["actual"]), r["status"], r["note"]])
    return buffer.getvalue()
