#!/usr/bin/env python3
"""Run the harness evaluation set against the model and the running rag-mcp.

  uv run --all-packages --frozen python scripts/evaluate_harness.py --label baseline --repeat 3
  uv run --all-packages --frozen python scripts/evaluate_harness.py --compare reports/eval/a.json reports/eval/b.json

Each run calls the text model once per case (synthetic copy only, no images). Reports go to reports/eval,
which Git ignores; paste the --compare table into the sibling backend/docs/13-harness-evaluation.md when a prompt,
limit or model changes.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import warnings

from local_config import ROOT, load_env

CASES = ROOT / "evals" / "2026-10-21" / "harness_cases.json"
PROMPT = ROOT / "services" / "agent" / "src" / "capstone_agent" / "prompts" / "text_review.md"
RULES = ROOT / "services" / "rag-mcp" / "src" / "capstone_rag_mcp" / "claim_rules.json"
REPORTS = ROOT / "reports" / "eval"


def git_commit() -> str:
    run = lambda *args: subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    commit = run("rev-parse", "--short", "HEAD") or "unknown"
    return commit + ("+dirty" if run("status", "--porcelain") else "")


async def run_once(data: dict, cases: list[dict], rule_sweep: bool = True) -> dict:
    from evaluation import score_case, summarize, texts_of
    from capstone_agent.regulations import legal_basis
    from capstone_agent.review_engine import Provider, review_text
    from capstone_agent.tools import McpTools

    provider, errors = Provider(), []
    if not provider.client:
        raise SystemExit("OPENAI_API_KEY is not set. The evaluation needs the text model.")
    started = time.monotonic()
    async with McpTools(servers=("rag",)) as call:
        async def one(case):
            texts = texts_of(case)
            findings, failures = await review_text(provider, texts)
            errors.extend(f"{case['id']}: {failure}" for failure in failures)
            # Without texts legal_basis skips the rule sweep, which measures what the sweep adds.
            lookup = await legal_basis(findings, texts=texts if rule_sweep else None, call=call)
            if lookup["status"] not in ("connected", "skipped"):
                errors.append(f"{case['id']}: legal basis {lookup['status']}")
            return score_case(case, lookup["findings"], data["basis_groups"])
        scored = await asyncio.gather(*(one(case) for case in cases))
    return {"seconds": round(time.monotonic() - started, 1), "errors": errors,
            "summary": summarize(scored), "cases": scored}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run", help="Name for this run, e.g. baseline or prompt-v2")
    parser.add_argument("--repeat", type=int, default=1, help="Runs to average; model output varies between runs")
    parser.add_argument("--case", action="append", help="Only this case id (repeatable)")
    parser.add_argument("--no-rule-sweep", action="store_true", help="Leave out the rule sweep to measure its share")
    parser.add_argument("--env-file")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()

    from evaluation import average, compare_table, render, validate
    if args.compare:
        before, after = (json.loads(p.read_text(encoding="utf-8")) for p in args.compare)
        print(compare_table(before, after))
        return

    data = json.loads(CASES.read_text(encoding="utf-8"))
    problems = validate(data)
    if problems:
        raise SystemExit("Evaluation set has invalid labels:\n" + "\n".join(problems))
    cases = [c for c in data["cases"] if not args.case or c["id"] in args.case]
    load_env(args.env_file)
    # LangSmith's OpenAI wrapper warns while serializing parsed responses; the results are unaffected.
    warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
    runs = [asyncio.run(run_once(data, cases, rule_sweep=not args.no_rule_sweep)) for _ in range(max(1, args.repeat))]
    checked = datetime.now(timezone.utc)
    report = {
        "config": {"label": args.label, "checkedAt": checked.isoformat(timespec="seconds"), "runs": len(runs),
                   "cases": len(cases), "casesVersion": data["version"],
                   "textModel": os.getenv("TEXT_MODEL", "gpt-5.4-mini"),
                   "fallbackTextModel": os.getenv("FALLBACK_TEXT_MODEL") or None,
                   "promptHash": hashlib.sha256(PROMPT.read_bytes()).hexdigest()[:8],
                   "rulesVersion": json.loads(RULES.read_text(encoding="utf-8"))["version"],
                   "ruleSweep": not args.no_rule_sweep,
                   "commit": git_commit()},
        "mean": average([run["summary"] for run in runs]),
        "runs": runs,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    stem = REPORTS / f"harness-{checked:%Y%m%d-%H%M%S}-{args.label}"
    stem.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stem.with_suffix(".md").write_text(render(report), encoding="utf-8")
    print(render(report))
    print(f"Saved {stem.relative_to(ROOT)}.json and .md")


if __name__ == "__main__":
    main()
