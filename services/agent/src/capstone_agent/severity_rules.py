"""Severity floors decided by public records, not by the model's reading of one phrase.

An ad that pictures a patent certificate KIPRIS lists as lapsed (소멸) presents a right the advertiser no longer has as
the basis of what it says. Every claim in that same ad file leans on it, so each is raised to high, with the reason.
Only claims in the file that shows the lapsed patent are raised; other files of the review keep their severity.
"""
from .patent_certificate import patent_numbers


def files_showing(numbers: list[str], texts: list[dict]) -> set[str]:
    digits = {n.split("-")[-1] for n in numbers if n}
    shown = set()
    for t in texts:
        found = {n.split("-")[-1] for n in patent_numbers(t.get("text", ""))}
        squashed = t.get("text", "").replace(" ", "")
        if digits & found or any(d and d in squashed for d in digits):
            shown.add((t.get("location") or {}).get("fileId"))
    return shown


def raise_for_lapsed_patents(findings: list[dict], texts: list[dict], patents: list[dict] | None) -> list[dict]:
    lapsed = [p.get("number") for p in patents or [] if isinstance(p, dict) and p.get("status") == "expired"]
    files = files_showing(lapsed, texts) if lapsed else set()
    if not files:
        return findings
    raised = []
    for f in findings:
        if (f.get("location") or {}).get("fileId") in files and f.get("severity") != "high":
            f = {**f, "escalation": {"from": f.get("severity"), "reason": f"광고가 근거로 내세운 특허({', '.join(lapsed)})가 소멸된 상태입니다. "
                                     "소멸된 특허를 내세운 광고의 주장은 높음으로 봅니다."}, "severity": "high"}
        raised.append(f)
    return raised
