"""Harness: Evaluation. Scores the text review step against a labelled set of synthetic ad copy.

The run goes through the same code as a review: review_text (text_review prompt, grounding filter, merge)
and legal_basis (rule sweep and basis search over rag-mcp). Images and OCR are left out, so a prompt, limit
or model change is measured on the claim extraction and the cited provisions alone.

Metrics, per run:
  recall         labelled claims that some finding covers (same line, overlapping phrase)
  precision      covering findings / (covering findings + findings on clean lines)
  clean_flagged  clean lines with at least one finding, the false positives a reviewer would see
  type_accuracy  covered claims whose finding has an accepted claimType
  basis_match    covered claims whose finding cites a provision from an accepted basis group
Lines without claims or clean are judgement calls and are not scored.
"""
from collections import Counter
import re

CLAIM_TYPES = {"wrinkle", "whitening", "clinical", "absolute", "ingredient", "patent", "exaggeration", "drug"}
METRICS = ("recall", "model_recall", "precision", "clean_flagged_rate", "type_accuracy", "basis_match")


def squeeze(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def validate(data: dict) -> list[str]:
    """Problems in the labelled set. An empty list means every label can be scored."""
    problems, ids = [], Counter(case["id"] for case in data["cases"])
    problems += [f"duplicate case id {i}" for i, n in ids.items() if n > 1]
    for case in data["cases"]:
        for n, line in enumerate(case["lines"]):
            where = f"{case['id']} line {n}"
            if line.get("clean") and line.get("claims"):
                problems.append(f"{where}: clean line has claims")
            for claim in line.get("claims", []):
                if squeeze(claim["phrase"]) not in squeeze(line["text"]):
                    problems.append(f"{where}: phrase {claim['phrase']!r} is not in the line")
                problems += [f"{where}: unknown type {t}" for t in claim["types"] if t not in CLAIM_TYPES]
                problems += [f"{where}: unknown basis group {g}" for g in claim["basis"] if g not in data["basis_groups"]]
    return problems


def texts_of(case: dict) -> list[dict]:
    """Ad copy lines in the shape extract_text returns."""
    return [{"id": f"{case['id']}:{n}", "text": line["text"], "location": {"kind": "copy"}}
            for n, line in enumerate(case["lines"])]


def covers(finding: dict, phrase: str) -> bool:
    found, wanted = squeeze(finding["phrase"]), squeeze(phrase)
    return bool(found) and (wanted in found or found in wanted)


def cites(finding: dict, groups: list[str], basis_groups: dict) -> bool:
    accepted = [ref for group in groups for ref in basis_groups[group]]
    return any(ref["source"] in (item.get("sourceName") or "") and (item.get("article") or "").startswith(ref["article"])
               for item in finding.get("basis", []) for ref in accepted)


def score_case(case: dict, findings: list[dict], basis_groups: dict) -> dict:
    claims, false_positives, clean_lines, matched = [], [], 0, set()
    for n, line in enumerate(case["lines"]):
        on_line = [f for f in findings if f.get("textId") == f"{case['id']}:{n}"]
        if line.get("clean"):
            clean_lines += 1
            false_positives += [{"line": line["text"], "phrase": f["phrase"], "claimType": f.get("claimType"),
                                 "source": f.get("source", "model")} for f in on_line]
            continue
        for claim in line.get("claims", []):
            hits = [f for f in on_line if covers(f, claim["phrase"])]
            matched.update(id(f) for f in hits)
            # A model finding is preferred over the rule sweep's whole-line finding when both cover the claim.
            best = next((f for f in hits if f.get("source") != "rule_sweep"), hits[0] if hits else None)
            claims.append({
                "line": line["text"], "phrase": claim["phrase"], "expectedTypes": claim["types"],
                "detected": best is not None,
                "by": None if best is None else best.get("source", "model"),
                "foundPhrase": best and best["phrase"], "foundType": best and best.get("claimType"),
                "typeOk": None if best is None else best.get("claimType") in claim["types"],
                "basisOk": None if best is None else any(cites(f, claim["basis"], basis_groups) for f in hits),
                "cited": [] if best is None else [f"{b.get('sourceName')} {b.get('article')}" for b in best.get("basis", [])],
            })
    return {"id": case["id"], "claims": claims, "falsePositives": false_positives, "cleanLines": clean_lines,
            "cleanFlagged": len({fp["line"] for fp in false_positives}), "coveringFindings": len(matched)}


def ratio(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def summarize(scored: list[dict]) -> dict:
    claims = [c for case in scored for c in case["claims"]]
    detected = [c for c in claims if c["detected"]]
    false_positives = sum(len(case["falsePositives"]) for case in scored)
    covering = sum(case["coveringFindings"] for case in scored)
    by_type: dict[str, dict] = {}
    for c in claims:
        row = by_type.setdefault(c["expectedTypes"][0], {"claims": 0, "detected": 0})
        row["claims"] += 1
        row["detected"] += c["detected"]
    return {
        "claims": len(claims), "detected": len(detected),
        "recall": ratio(len(detected), len(claims)),
        "model_recall": ratio(sum(c["by"] == "model" for c in detected), len(claims)),
        "precision": ratio(covering, covering + false_positives),
        "false_positives": false_positives,
        "clean_lines": sum(case["cleanLines"] for case in scored),
        "clean_flagged": sum(case["cleanFlagged"] for case in scored),
        "clean_flagged_rate": ratio(sum(case["cleanFlagged"] for case in scored), sum(case["cleanLines"] for case in scored)),
        "type_accuracy": ratio(sum(bool(c["typeOk"]) for c in detected), len(detected)),
        "basis_match": ratio(sum(bool(c["basisOk"]) for c in detected), len(detected)),
        "by_type": {k: {**v, "recall": ratio(v["detected"], v["claims"])} for k, v in sorted(by_type.items())},
    }


def average(summaries: list[dict]) -> dict:
    """Mean of each metric over repeated runs. Model output varies between runs, so one run is not a result."""
    mean = {}
    for key in METRICS:
        values = [s[key] for s in summaries if s[key] is not None]
        mean[key] = round(sum(values) / len(values), 4) if values else None
    mean["false_positives"] = round(sum(s["false_positives"] for s in summaries) / len(summaries), 2)
    mean["runs"] = len(summaries)
    return mean


def percent(value) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


LABELS = {"recall": "탐지율", "model_recall": "모델 단독 탐지율", "precision": "정밀도",
          "clean_flagged_rate": "오탐 줄 비율", "type_accuracy": "유형 일치", "basis_match": "근거 조항 일치"}


def compare_table(before: dict, after: dict) -> str:
    """Table of two runs for the evaluation log (backend/docs/13-harness-evaluation/2026-10-21.md), headed by their labels."""
    names = before["config"]["label"], after["config"]["label"]
    rows = [f"| 지표 | {names[0]} | {names[1]} | 차이 |", "| --- | --- | --- | --- |"]
    for key in METRICS:
        b, a = before["mean"][key], after["mean"][key]
        delta = "-" if a is None or b is None else f"{(a - b) * 100:+.1f}%p"
        rows.append(f"| {LABELS[key]} | {percent(b)} | {percent(a)} | {delta} |")
    rows.append(f"| 오탐 건수(평균) | {before['mean']['false_positives']} | {after['mean']['false_positives']} | "
                f"{after['mean']['false_positives'] - before['mean']['false_positives']:+.2f} |")
    config = ["", f"| 설정 | {names[0]} | {names[1]} |", "| --- | --- | --- |"]
    for key in ("textModel", "promptHash", "rulesVersion", "ruleSweep", "casesVersion", "commit", "runs"):
        config.append(f"| {key} | {before['config'].get(key)} | {after['config'].get(key)} |")
    return "\n".join(rows + config)


def render(report: dict) -> str:
    mean, config, last = report["mean"], report["config"], report["runs"][-1]
    lines = [f"# 하네스 평가 {config['label']}", "",
             f"- 실행 {config['checkedAt']}, 반복 {config['runs']}회, 평가셋 {config['casesVersion']}",
             f"- 모델 {config['textModel']}, 프롬프트 {config['promptHash']}, 규칙 {config['rulesVersion']}, "
             f"규칙 sweep {'켬' if config.get('ruleSweep', True) else '끔'}, 커밋 {config['commit']}",
             "", "| 지표 | 값 |", "| --- | --- |"]
    lines += [f"| {LABELS[k]} | {percent(mean[k])} |" for k in METRICS]
    lines += [f"| 오탐 건수(평균) | {mean['false_positives']} |", "",
              "## 유형별 탐지율 (마지막 실행)", "", "| 정답 유형 | 정답 | 탐지 | 탐지율 |", "| --- | --- | --- | --- |"]
    lines += [f"| {t} | {v['claims']} | {v['detected']} | {percent(v['recall'])} |" for t, v in last["summary"]["by_type"].items()]
    missed = [(case["id"], c) for case in last["cases"] for c in case["claims"] if not c["detected"]]
    wrong_basis = [(case["id"], c) for case in last["cases"] for c in case["claims"] if c["detected"] and not c["basisOk"]]
    false_positives = [(case["id"], fp) for case in last["cases"] for fp in case["falsePositives"]]
    lines += ["", "## 놓친 주장 (마지막 실행)", ""]
    lines += [f"- {cid}: \"{c['line']}\" 중 \"{c['phrase']}\"" for cid, c in missed] or ["- 없음"]
    lines += ["", "## 근거 조항이 맞지 않은 주장 (마지막 실행)", ""]
    lines += [f"- {cid}: \"{c['phrase']}\" 인용 {', '.join(c['cited']) or '없음'}" for cid, c in wrong_basis] or ["- 없음"]
    lines += ["", "## 오탐 (마지막 실행)", ""]
    lines += [f"- {cid}: \"{fp['line']}\" 에서 \"{fp['phrase']}\" ({fp['claimType']}, {fp['source']})"
              for cid, fp in false_positives] or ["- 없음"]
    errors = [e for run in report["runs"] for e in run["errors"]]
    if errors:
        lines += ["", "## 실행 오류", ""] + [f"- {e}" for e in errors]
    return "\n".join(lines) + "\n"
