import json
from pathlib import Path

from capstone_agent.evaluation import average, compare_table, score_case, summarize, validate

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"drug": [{"source": "화장품법", "article": "제13조"}]}
CASE = {"id": "c", "lines": [
    {"text": "아토피 치료 크림", "claims": [{"phrase": "치료", "types": ["drug"], "basis": ["drug"]}]},
    {"text": "48시간 보습", "claims": [{"phrase": "48시간", "types": ["clinical"], "basis": ["drug"]}]},
    {"text": "산뜻한 향", "clean": True},
    {"text": "판단이 갈리는 줄"},
]}


def finding(line, phrase, claim_type="drug", source=None, basis=()):
    f = {"textId": f"c:{line}", "phrase": phrase, "claimType": claim_type, "basis": list(basis)}
    return {**f, "source": source} if source else f


def test_labelled_set_is_scorable():
    data = json.loads((ROOT / "evals" / "harness_cases.json").read_text(encoding="utf-8"))
    assert validate(data) == []
    assert len(data["cases"]) >= 20


def test_validate_reports_a_phrase_missing_from_its_line():
    bad = {"basis_groups": GROUPS, "cases": [{"id": "x", "lines": [
        {"text": "촉촉한 크림", "claims": [{"phrase": "치료", "types": ["drug"], "basis": ["drug"]}]}]}]}
    assert any("not in the line" in p for p in validate(bad))


def test_scores_detection_type_basis_and_false_positives():
    findings = [
        finding(0, "아토피 치료", basis=[{"sourceName": "화장품법", "article": "제13조 (부당한 표시ㆍ광고 행위 등의 금지) 제1항제1호"}]),
        finding(2, "산뜻한 향", "exaggeration"),
        finding(3, "판단이 갈리는 줄", "exaggeration"),  # unscored line
    ]
    scored = score_case(CASE, findings, GROUPS)
    first, second = scored["claims"]
    assert first["detected"] and first["typeOk"] and first["basisOk"] and first["by"] == "model"
    assert not second["detected"] and second["typeOk"] is None
    assert scored["cleanFlagged"] == 1 and len(scored["falsePositives"]) == 1

    summary = summarize([scored])
    assert summary["recall"] == 0.5 and summary["precision"] == 0.5
    assert summary["clean_flagged_rate"] == 1.0 and summary["basis_match"] == 1.0


def test_rule_sweep_whole_line_counts_but_model_finding_is_preferred():
    findings = [finding(1, "48시간 보습", "clinical", source="rule_sweep"), finding(1, "48시간", "exaggeration")]
    claim = score_case(CASE, findings, GROUPS)["claims"][1]
    assert claim["by"] == "model" and claim["typeOk"] is False
    only_sweep = score_case(CASE, findings[:1], GROUPS)
    assert only_sweep["claims"][1]["by"] == "rule_sweep"
    assert summarize([only_sweep])["model_recall"] == 0.0


def test_compare_table_shows_the_change_between_runs():
    one = summarize([score_case(CASE, [finding(0, "치료")], GROUPS)])
    two = summarize([score_case(CASE, [finding(0, "치료"), finding(1, "48시간", "clinical")], GROUPS)])
    config = {"label": "a", "textModel": "m", "promptHash": "p", "rulesVersion": "r", "casesVersion": "v", "commit": "c", "runs": 1}
    table = compare_table({"mean": average([one]), "config": config}, {"mean": average([two]), "config": {**config, "label": "b"}})
    assert "| 지표 | a | b | 차이 |" in table
    assert "| 탐지율 | 50.0% | 100.0% | +50.0%p |" in table
