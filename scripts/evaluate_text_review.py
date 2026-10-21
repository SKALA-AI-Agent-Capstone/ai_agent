"""Run a small, explicitly synthetic text-review set against the configured model.

Usage: uv run --all-packages --frozen python scripts/evaluate_text_review.py
This makes paid model calls and prints aggregate metrics without raw ad text.
"""

import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import re
import time

from local_config import load_env

from capstone_agent.context_budget import split_parse_text
from capstone_agent.instructions import instruction
from capstone_agent.review_engine import Provider
from capstone_agent.review_models import TextAssessment


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "tests/data/text_review_eval.json"


def materialize(case):
    return (case.get("prefix", "") + case.get("text", "") * case.get("repeat_before", 0)
            + case.get("insert", "") + case.get("text", "") * case.get("repeat_after", 0)
            + case.get("suffix", ""))


def expected_spans(text, phrases):
    return {(match.start(), phrase) for phrase in set(phrases)
            for match in re.finditer(re.escape(phrase), text)}


async def evaluate(provider, cases):
    rows = []
    for case in cases:
        ad = materialize(case)
        chunks = split_parse_text(instruction("text_review"), ad, TextAssessment)
        predicted = set()
        started = time.monotonic()
        for start, chunk in chunks:
            output = await provider.parse(instruction("text_review"), TextAssessment, text=chunk)
            used = Counter()
            for finding in output.findings:
                positions = [match.start() for match in re.finditer(re.escape(finding.phrase), chunk)]
                index = used[finding.phrase]
                used[finding.phrase] += 1
                if index < len(positions):
                    predicted.add((start + positions[index], finding.phrase))
        expected = expected_spans(ad, case["expected"])
        rows.append({"id": case["id"], "chunks": len(chunks), "tp": len(predicted & expected),
                     "fp": len(predicted - expected), "fn": len(expected - predicted),
                     "seconds": round(time.monotonic() - started, 2)})
    counts = {key: sum(row[key] for row in rows) for key in ("tp", "fp", "fn")}
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else None
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else None
    return {"dataset": "synthetic_text_review_v1", "cases": rows, **counts,
            "precision": precision, "recall": recall,
            "note": "Synthetic regression set; expert-labeled real advertisements are still needed."}


async def main():
    load_env(None)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for the live evaluation")
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    result = await evaluate(Provider(), cases)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
