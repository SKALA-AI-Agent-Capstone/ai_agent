#!/usr/bin/env python3
"""Generate JSON Schemas from shared Pydantic models."""
import argparse
import json
from pathlib import Path

from capstone_shared.rag_contracts import ClaimRulesResult, PrecedentSearchResult, RegulationSearchResult
from capstone_shared.review_contracts import RunRequest, ResumeRequest, RewriteRequest, RewriteSuggestion, VerifyRewriteRequest, RewriteVerification, TextAssessment, VisualAssessment, ResultTile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
for name, model in [
    ("search-regulations.output.schema.json", RegulationSearchResult),
    ("search-precedents.output.schema.json", PrecedentSearchResult),
    ("lookup-claim-rules.output.schema.json", ClaimRulesResult),
    ("mvp-agent-run.input.schema.json", RunRequest),
    ("mvp-agent-resume.input.schema.json", ResumeRequest),
    ("mvp-agent-rewrite.input.schema.json", RewriteRequest),
    ("mvp-agent-rewrite.output.schema.json", RewriteSuggestion),
    ("mvp-agent-rewrite-verify.input.schema.json", VerifyRewriteRequest),
    ("mvp-agent-rewrite-verify.output.schema.json", RewriteVerification),
    ("mvp-text-assessment.schema.json", TextAssessment),
    ("mvp-visual-assessment.schema.json", VisualAssessment),
    ("mvp-result-tile.schema.json", ResultTile),
]:
    target = root / "contracts" / name
    content = json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not target.exists() or target.read_text(encoding="utf-8") != content:
            raise SystemExit(f"Contract schema is stale: {name}")
    else:
        target.parent.mkdir(exist_ok=True)
        target.write_text(content, encoding="utf-8")
print("Contract schemas are current.")
