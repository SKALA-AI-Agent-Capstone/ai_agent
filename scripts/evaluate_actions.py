#!/usr/bin/env python3
"""Run the harness evaluation set 2026-10-22 and write its validation report.

  uv run --all-packages --frozen python scripts/evaluate_actions.py --label baseline --repeat 3
  uv run --all-packages --frozen python scripts/evaluate_actions.py --label after-fix --repeat 3 --before reports/eval/<baseline>.json
  uv run --all-packages --frozen python scripts/evaluate_actions.py --case P01 --case P02

Needs the running rag-mcp and business-mcp and OPENAI_API_KEY. Evidence PDFs are read from evals/2026-10-22/fixtures
through the real extraction function; only the file transfer from the Backend is replaced. A case with
fault evidence_parse_error gets an injected parser failure. Everything goes to reports/eval (Git ignores it):
the raw report (.json), the validation report section (.md) and its rows as a CSV. Paste the .md into the
검증 결과 section of backend/docs/13-harness-evaluation/2026-10-22.md; --before fills 개선 전 and 개선 후.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch
import warnings

import httpx

from local_config import ROOT, load_env

CASES = ROOT / "evals" / "2026-10-22" / "harness_cases.json"
PROMPT = ROOT / "services" / "agent" / "src" / "capstone_agent" / "prompts" / "text_review.md"
RULES = ROOT / "services" / "rag-mcp" / "src" / "capstone_rag_mcp" / "claim_rules.json"
ACTIONS_JS = ROOT.parent / "frontend" / "src" / "shared" / "constants" / "reviewActions.js"
REPORTS = ROOT / "reports" / "eval"
NO_EVIDENCE = {"claimTypes": [], "requirements": [], "flags": [], "status": "not_submitted"}


def git_commit() -> str:
    run = lambda *args: subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return (run("rev-parse", "--short", "HEAD") or "unknown") + ("+dirty" if run("status", "--porcelain") else "")


def action_for(findings: list[dict]) -> list[str]:
    """The frontend's actionFor, run in Node so the evaluation scores the rule the screen uses."""
    source = "\n".join(line for line in ACTIONS_JS.read_text(encoding="utf-8").splitlines() if not line.startswith("import "))
    script = ("const t = (key) => key;\nimport fs from 'node:fs';\n" + source +
              "\nconsole.log(JSON.stringify(JSON.parse(fs.readFileSync(0, 'utf8')).map(actionFor)));")
    done = subprocess.run(["node", "--input-type=module", "-e", script], input=json.dumps(findings),
                          text=True, capture_output=True, check=True)
    return json.loads(done.stdout)


async def extract_evidence(case: dict) -> tuple[dict, int]:
    """Run the business-mcp extraction on the case's PDFs. Called one case at a time, before the model runs,
    because it swaps the HTTP fetch for a local file read."""
    from capstone_agent.guardrails import mask_tool_result
    from capstone_business_mcp import review_tools

    files = case.get("evidenceFiles") or []
    if not files:
        return dict(NO_EVIDENCE), 0
    paths = [CASES.parent / f for f in files]
    documents = [{"id": f"{case['id']}-{n}", "name": p.name, "docType": "test_report", "mimeType": "application/pdf"}
                 for n, p in enumerate(paths)]
    by_id = {d["id"]: p.read_bytes() for d, p in zip(documents, paths)}

    async def fetch(self, url, *args, **kwargs):
        content = by_id[url.rsplit("/", 1)[-1]]
        return httpx.Response(200, content=content, request=httpx.Request("GET", url))

    faults = [patch.object(review_tools, "prepare_evidence", side_effect=RuntimeError("injected parser failure"))] \
        if case.get("fault") == "evidence_parse_error" else []
    with patch.object(httpx.AsyncClient, "get", fetch):
        for fault in faults:
            fault.start()
        try:
            evidence = await review_tools.structure("evaluation", case["id"], documents, 1, "evaluation-only")
        finally:
            for fault in faults:
                fault.stop()
    # The review passes tool results through the same masking before a model sees them.
    return mask_tool_result("structure_evidence", evidence), len(files)


async def run_case(provider, rag, case: dict, evidence: dict, evidence_count: int, today: str) -> dict:
    from action_evaluation import case_action, case_texts, parse_condition
    from capstone_agent.checks import cross_checks
    from capstone_agent.evidence import match as match_evidence, summarize
    from capstone_agent.regulations import legal_basis
    from capstone_agent.review_engine import assess_severity, review_text

    texts, product = case_texts(case), parse_condition(case.get("condition", ""))
    errors = []
    try:
        findings, failures = await review_text(provider, texts)
        errors += failures
        lookup = await legal_basis(findings, texts=texts, call=rag)
        if lookup["status"] not in ("connected", "skipped"):
            errors.append(f"LEGAL_BASIS_{lookup['status'].upper()}")
        findings, evidence, matches = lookup["findings"], dict(evidence), {}
        if evidence.get("documents"):
            try:
                evidence["summaries"] = await summarize(provider, evidence["documents"])
                matches = await match_evidence(provider, findings, evidence["summaries"], [t["text"] for t in texts])
            except Exception:
                errors.append("EVIDENCE_MATCH_FAILED")
        checked = await cross_checks(findings, product=product, evidence=evidence, evidence_count=evidence_count,
                                     ad_text="\n".join(t["text"] for t in texts), provider=provider,
                                     review_date=today, matches=matches)
        errors += [w for w in checked["warnings"] if w == "CROSS_CHECK_UNAVAILABLE"]
        findings = checked["findings"]
        severities = await asyncio.gather(*(assess_severity(provider, f, evidence.get("text", "")) for f in findings))
        for finding, (severity, _) in zip(findings, severities):
            finding["severity"] = severity
        actions = action_for(findings) if findings else []
    except Exception as error:
        return {"id": case["id"], "actual": None, "error": type(error).__name__, "cards": []}
    return {"id": case["id"], "actual": case_action(actions), "error": ", ".join(errors) or None,
            "evidenceStatus": evidence.get("status"),
            "cards": [{"phrase": f["phrase"], "action": a, "severity": f.get("severity"), "claimType": f.get("claimType"),
                       "rules": [r["id"] for r in f.get("rules", [])],
                       "checks": [{"type": c.get("type"), "status": c.get("status")} for c in f.get("checks", [])]}
                      for f, a in zip(findings, actions)]}


async def run_once(cases: list[dict], evidence: dict, today: str) -> list[dict]:
    from capstone_agent.review_engine import Provider
    from capstone_agent.tools import McpTools

    provider = Provider()
    if not provider.client:
        raise SystemExit("OPENAI_API_KEY is not set. The evaluation needs the text model.")
    async with McpTools(servers=("rag",)) as rag:
        limit = asyncio.Semaphore(4)

        async def one(case):
            async with limit:
                return await run_case(provider, rag, case, *evidence[case["id"]], today)
        return await asyncio.gather(*(one(case) for case in cases))


def evidence_warnings(evidence: dict) -> dict:
    """Extraction failures per case, e.g. OCR_UNAVAILABLE p.28 for a scanned page."""
    out = {}
    for case_id, (extracted, _) in evidence.items():
        codes = [f"{f['code']}" + (f" p.{f['page']}" if f.get("page") else "") for f in extracted.get("failures", [])]
        if codes:
            out[case_id] = codes
    return out


def prompt_date() -> str | None:
    """Date of the last commit that changed the text_review prompt, shown as the prompt version."""
    done = subprocess.run(["git", "log", "-1", "--format=%cs", "--", str(PROMPT)], cwd=ROOT, capture_output=True, text=True)
    return done.stdout.strip() or None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run")
    parser.add_argument("--repeat", type=int, default=1, help="Runs to compare; model output varies between runs")
    parser.add_argument("--case", action="append", help="Only this case id (repeatable)")
    parser.add_argument("--before", type=Path, help="Earlier report (.json) to compare against in the result line")
    parser.add_argument("--env-file")
    args = parser.parse_args()

    from action_evaluation import render, summarize, to_csv, validate
    data = json.loads(CASES.read_text(encoding="utf-8"))
    problems = validate(data)
    if problems:
        raise SystemExit("Evaluation set has invalid cases:\n" + "\n".join(problems))
    cases = [c for c in data["cases"] if not args.case or c["id"] in args.case]
    missing = [f for c in cases for f in c.get("evidenceFiles") or [] if not (CASES.parent / f).is_file()]
    if missing:
        raise SystemExit("Evidence files are missing (see evals/2026-10-22/fixtures/README.md): " + ", ".join(sorted(set(missing))))
    before = json.loads(args.before.read_text(encoding="utf-8")) if args.before else None
    load_env(args.env_file)
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

    checked = datetime.now(timezone.utc)
    today = checked.date().isoformat()
    evidence = {}
    for case in cases:
        evidence[case["id"]] = asyncio.run(extract_evidence(case))
    runs = [asyncio.run(run_once(cases, evidence, today)) for _ in range(max(1, args.repeat))]

    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"actions-{checked:%Y%m%d-%H%M%S}-{args.label}.json"
    report = {
        "config": {"label": args.label, "checkedAt": checked.isoformat(timespec="seconds"), "runs": len(runs),
                   "cases": len(cases), "casesVersion": data["version"],
                   "textModel": os.getenv("TEXT_MODEL", "gpt-5.4-mini"),
                   "promptHash": hashlib.sha256(PROMPT.read_bytes()).hexdigest()[:8],
                   "promptDate": prompt_date(),
                   "rulesVersion": json.loads(RULES.read_text(encoding="utf-8"))["version"],
                   "commit": git_commit(), "reportPath": str(path.relative_to(ROOT)),
                   "evidenceWarnings": evidence_warnings(evidence)},
        "summary": summarize(cases, runs),
        "runs": runs,
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    section = render(report, cases, before)
    path.with_suffix(".md").write_text(section, encoding="utf-8")
    path.with_suffix(".csv").write_text(to_csv(report, cases), encoding="utf-8-sig")
    print(section)
    print(f"Saved {path.relative_to(ROOT)} with .md and .csv")


if __name__ == "__main__":
    main()
